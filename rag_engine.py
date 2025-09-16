# rag_engine.py
import os
import re
import tempfile
from typing import List, Tuple, Sequence

import numpy as np
from dotenv import load_dotenv

from llama_index.core import (
    VectorStoreIndex,
    Settings,
    PromptTemplate,
    Document,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PyMuPDFReader
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.node_parser import TokenTextSplitter

# ---- version-proof retriever imports ----
from llama_index.core.retrievers import VectorIndexRetriever

# BM25 lives in a split pkg on newer versions
BM25_AVAILABLE = True
try:
    from llama_index.retrievers.bm25 import BM25Retriever  # 0.14+
except Exception:
    try:
        from llama_index.core.retrievers.bm25 import BM25Retriever  # older
    except Exception:
        BM25_AVAILABLE = False
        BM25Retriever = None  # type: ignore

# QueryFusion path differs across versions
try:
    from llama_index.core.retrievers import QueryFusionRetriever
except Exception:
    from llama_index.core.retrievers.fusion import QueryFusionRetriever  # type: ignore

from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.llms.ollama import Ollama
from qdrant_client import QdrantClient

load_dotenv()

# --- Env config (safe defaults) ---
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

TOP_K = int(os.getenv("TOP_K", "12"))           # total candidates considered
FUSION_K = int(os.getenv("FUSION_K", "6"))      # top-k per retriever pre-fusion
NUM_PREDICT = int(os.getenv("NUM_PREDICT", "600"))
NUM_CTX = int(os.getenv("NUM_CTX", "4096"))

# BM25 feature toggle (set USE_BM25=0 to disable quickly)
USE_BM25 = os.getenv("USE_BM25", "1") != "0"

# Dynamic grounding controls
MIN_EVID_SENTENCES = int(os.getenv("MIN_EVID_SENTENCES", "2"))
PERCENTILE_GATE = float(os.getenv("PERCENTILE_GATE", "0.80"))  # keep >= 80th percentile
ABS_MIN_SIM = float(os.getenv("ABS_MIN_SIM", "0.12"))

# Cross-encoder reranker
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "12"))


class RAGEngine:
    def __init__(self):
        # Embeddings (multilingual E5) with cache
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="intfloat/multilingual-e5-small",
            cache_folder=os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"),
        )

        # LLM via Ollama (deterministic + streaming)
        Settings.llm = Ollama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            streaming=True,
            request_timeout=800.0,
            temperature=0,
            keep_alive="10m",
            additional_kwargs={
                "num_predict": NUM_PREDICT,
                "num_ctx": NUM_CTX,
            },
        )
        print(f"🔍 LLM in use: {OLLAMA_MODEL} @ {OLLAMA_BASE_URL}")

        # Vector DB (Qdrant)
        self.qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client, collection_name="agentic_rag_collection"
        )

        # Optional grounded memory (bias only; never cited)
        self.memory_store = QdrantVectorStore(
            client=self.qdrant_client, collection_name="agentic_rag_memory"
        )
        self.memory_index = VectorStoreIndex.from_documents([], vector_store=self.memory_store)
        self.use_grounded_memory = True
        self.memory_doc_count = 0
        self.max_memory_docs = 200
        self._last_memory_text = None

        # Indices & retrievers
        self.index = None
        self.retriever_vec = None
        self.retriever_bm25 = None
        self.retriever = None  # fusion or vector-only

        # Reranker
        self.reranker = SentenceTransformerRerank(
            top_n=RERANK_TOP_N,
            model=RERANK_MODEL,
        )

        # Retrieval bias
        self._last_cited_files: set[str] = set()

        # Chat history (intent only)
        self.chat_history = []

        # QA prompt
        self.qa_template = PromptTemplate(
            "Answer the question using ONLY the context below. "
            "If the answer is not fully supported by the context, reply exactly: No answer found.\n\n"
            "Conversation (for intent only; do NOT use for facts):\n{chat_window}\n\n"
            "Context:\n{context_str}\n\n"
            "Question: {query_str}\n"
            "Answer:"
        )

    # ---------- Document indexing ----------
    def load_documents(self, uploaded_files):
        reader = PyMuPDFReader()
        documents = []

        with tempfile.TemporaryDirectory() as tmpdir:
            for file in uploaded_files:
                path = os.path.join(tmpdir, file.name)
                with open(path, "wb") as f:
                    f.write(file.getbuffer())
                documents.extend(reader.load_data(path))  # keeps file_name/page metadata

        # Slightly smaller chunks tend to boost recall; overlap preserves continuity
        node_parser = TokenTextSplitter(chunk_size=800, chunk_overlap=120)

        self.index = VectorStoreIndex.from_documents(
            documents, vector_store=self.vector_store, node_parser=node_parser
        )

        # Vector retriever
        self.retriever_vec = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=max(FUSION_K, 6),
        )

        # Try BM25; skip gracefully if not available or disabled
        self.retriever_bm25 = None
        if BM25_AVAILABLE and USE_BM25:
            try:
                self.retriever_bm25 = BM25Retriever.from_defaults(
                    docstore=self.index.docstore,
                    similarity_top_k=max(FUSION_K, 6),
                )
            except Exception as e:
                print(f"[BM25] disabled: {e}")
                self.retriever_bm25 = None

        # Build final retriever (fusion if both available)
        retrievers = [self.retriever_vec] + ([self.retriever_bm25] if self.retriever_bm25 else [])
        if len(retrievers) == 1:
            self.retriever = self.retriever_vec
        else:
            self.retriever = QueryFusionRetriever(
                retrievers=retrievers,
                num_queries=3,             # HyDE/paraphrases improve coverage
                mode="reciprocal_rerank",  # stable, strong default
                llm=Settings.llm,
                with_context=True,
                use_async=False,
                similarity_top_k=TOP_K,
            )

    # ---------- Helpers ----------
    @staticmethod
    def _format_chat_window(history, k=6):
        if not history:
            return "(empty)"
        window = history[-k:]
        return "\n".join(f"{m.get('role','user').capitalize()}: {m.get('content','')}" for m in window)

    @staticmethod
    def _get_text(node):
        t = getattr(node, "text", None)
        if not t and hasattr(node, "get_content"):
            try:
                t = node.get_content()
            except Exception:
                t = None
        return (t or "").strip()

    @staticmethod
    def _unit_normalize(mat: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms

    def _context_from_nodes(self, nodes_docs) -> str:
        parts = []
        for n in nodes_docs:
            meta = getattr(n, "metadata", {}) or {}
            file = meta.get("file_name", "Unknown")
            page = meta.get("page_label", meta.get("page_number"))
            page_str = f"{page}" if page is not None else "?"
            text = self._get_text(n)
            parts.append(f"--- {file}, page {page_str} ---\n{text}")
        return "\n\n".join(parts)

    def _bias_nodes_to_last_files(self, nodes, min_keep: int = 4):
        if not self._last_cited_files:
            return list(nodes)
        prefer, others = [], []
        for n in nodes:
            meta = getattr(n, "metadata", {}) or {}
            file = (meta.get("file_name") or "").lower()
            (prefer if file in self._last_cited_files else others).append(n)
        return prefer + others if len(prefer) >= min_keep else prefer + others[: max(0, min_keep - len(prefer))]

    # ---------- Deterministic citation post-processing ----------
    @staticmethod
    def _sentences(text: str) -> List[str]:
        chunks = re.split(r"(?<=[.!?])\s+", (text or "").strip())
        return [s.strip() for s in chunks if s.strip()]

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        embs = []
        for t in texts:
            try:
                embs.append(Settings.embed_model.get_text_embedding(t))
            except Exception:
                embs.append([0.0])
        maxd = max(len(e) for e in embs)
        embs = [np.pad(np.array(e, dtype="float32"), (0, maxd - len(e))) for e in embs]
        M = np.vstack(embs)
        return self._unit_normalize(M)

    def _attach_citations(self, answer_text: str, nodes_docs: list) -> Tuple[str, List[Tuple[int, str]]]:
        sentences = self._sentences(answer_text)
        if not sentences:
            return "No answer found.", []

        doc_texts, doc_meta = [], []
        for n in nodes_docs:
            t = self._get_text(n)
            if not t:
                continue
            meta = getattr(n, "metadata", {}) or {}
            file = meta.get("file_name", "Unknown")
            page = meta.get("page_label", meta.get("page_number"))
            page_str = f"{page}" if page is not None else "?"
            doc_texts.append(t)
            doc_meta.append(f"{file}, page {page_str}")

        if not doc_texts:
            return "No answer found.", []

        s_emb = self._embed(sentences)     # [S x d], unit normalized
        d_emb = self._embed(doc_texts).T   # [d x D]
        sims = s_emb @ d_emb               # cosine sims in [-1,1]

        flat = sims.flatten()
        p80 = np.percentile(flat, int(PERCENTILE_GATE * 100))
        gate = max(ABS_MIN_SIM, float(p80))

        used_map, used_order = {}, []
        out_sentences = []

        for i, s in enumerate(sentences):
            if len(re.findall(r"[A-Za-z0-9]", s)) < 6:
                continue
            j = int(np.argmax(sims[i]))
            score = float(sims[i, j])
            if score < gate:
                continue
            if j not in used_map:
                used_map[j] = len(used_map) + 1
                used_order.append(j)
            k = used_map[j]
            out_sentences.append(f"{s} [{k}]")

        if len(out_sentences) < MIN_EVID_SENTENCES:
            # insufficient evidence → let caller show top passages instead
            return "", [(i + 1, m) for i, m in enumerate(doc_meta[:3])]

        new_text = " ".join(out_sentences)
        sources = [(used_map[j], doc_meta[j]) for j in used_order]
        return new_text, sources

    # ---------- Grounded memory ----------
    def _save_memory(self, question: str, answer_text: str, sources: List[Tuple[int, str]]):
        if not answer_text or answer_text.strip().lower() == "no answer found.":
            return
        if not sources:
            return
        srcs = "; ".join(f"[{k}] {v}" for k, v in sources)
        blob = f"Q: {question}\nA: {answer_text}\nSources: {srcs}".strip()
        self.memory_index.insert(Document(text=blob, metadata={"source": "memory"}))
        self.memory_doc_count += 1
        self._last_memory_text = blob

        if self.memory_doc_count > self.max_memory_docs:
            self.memory_index = VectorStoreIndex.from_documents([], vector_store=self.memory_store)
            self.memory_doc_count = 0
            self._last_memory_text = None

    # ---------- Chat query ----------
    def query(self, question: str):
        if self.retriever is None:
            return {"answer": ["No documents indexed yet."], "sources": [], "messages": self.chat_history}

        # Retrieve candidates (hybrid if available)
        try:
            nodes_docs = list(self.retriever.retrieve(question)) or []
        except Exception:
            nodes_docs = []

        # Cross-encoder rerank
        if nodes_docs:
            try:
                nodes_docs = self.reranker.postprocess_nodes(nodes_docs, query_str=question)
            except Exception:
                pass

        nodes_docs = self._bias_nodes_to_last_files(nodes_docs)

        if not nodes_docs:
            self.chat_history.append({"role": "user", "content": question})
            self.chat_history.append({"role": "assistant", "content": "No answer found."})
            return {"answer": ["No answer found."], "sources": [], "messages": self.chat_history}

        # Build context & prompt
        context_str = self._context_from_nodes(nodes_docs[:TOP_K])
        chat_window = self._format_chat_window(self.chat_history, k=6)

        final_prompt = self.qa_template.format(
            context_str=context_str,
            query_str=question,
            chat_window=chat_window,
        )

        # LLM answer
        chunks = []
        for part in Settings.llm.stream_complete(final_prompt):
            delta = getattr(part, "delta", None)
            text = delta if isinstance(delta, str) else getattr(part, "text", "")
            if text:
                chunks.append(text)
        raw_answer = "".join(chunks).strip()

        # Attach citations, or fall back to passages
        full_text, ordered_sources = self._attach_citations(raw_answer, nodes_docs[:TOP_K])
        if not full_text:
            best = nodes_docs[:3]
            bullets = []
            for i, n in enumerate(best, 1):
                meta = getattr(n, "metadata", {}) or {}
                file = meta.get("file_name", "Unknown")
                page = meta.get("page_label", meta.get("page_number"))
                text = (self._get_text(n) or "").strip()
                text = re.sub(r"\s+", " ", text)
                snippet = text[:600] + ("…" if len(text) > 600 else "")
                bullets.append(f"[{i}] {file}, page {page} — {snippet}")
            full_text = "No answer found. Here are the most relevant passages:\n\n" + "\n\n".join(bullets)
            ordered_sources = [
                (i, f"{getattr(n,'metadata',{}).get('file_name','Unknown')}, page "
                    f"{getattr(n,'metadata',{}).get('page_label', getattr(n,'metadata',{}).get('page_number'))}")
                for i, n in enumerate(best, 1)
            ]

        # Update history & (optional) memory
        self.chat_history.append({"role": "user", "content": question})
        self.chat_history.append({"role": "assistant", "content": full_text})
        if not full_text.startswith("No answer found."):
            self._save_memory(question, full_text, ordered_sources)

        # Bias next retrieval
        cited_files = {src.split(",")[0].strip().lower() for _, src in ordered_sources}
        self._last_cited_files = cited_files

        return {
            "answer": [full_text],
            "sources": ordered_sources,
            "messages": self.chat_history,
        }
