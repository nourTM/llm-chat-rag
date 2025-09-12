# rag_engine.py

import os
import re
import tempfile
from typing import List, Tuple

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
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.ollama import Ollama
from llama_index.core.node_parser import TokenTextSplitter  # avoids NLTK punkt issues

from qdrant_client import QdrantClient

load_dotenv()

# --- Env config (with safe defaults for local dev) ---
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")         # must exist in `ollama list`
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Tuning knobs
TOP_K = int(os.getenv("TOP_K", "12"))               # how many chunks to retrieve
NUM_PREDICT = int(os.getenv("NUM_PREDICT", "600"))  # max output tokens
NUM_CTX = int(os.getenv("NUM_CTX", "4096"))         # context window
SIM_THRESHOLD = float(os.getenv("SIM_THRESHOLD", "0.18"))  # sentence→chunk grounding threshold


class _SimpleNode:
    """Minimal node to inject last grounded memory if you want to bias retrieval."""
    def __init__(self, text: str, metadata: dict | None = None):
        self.text = text
        self.metadata = metadata or {}


class RAGEngine:
    def __init__(self):
        # Embeddings (multilingual E5)
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
                # "num_thread": 8,  # uncomment & tune on CPU-only boxes if desired
            },
        )
        print(f"🔍 LLM in use: {OLLAMA_MODEL} @ {OLLAMA_BASE_URL}")

        # Vector DB (Qdrant)
        self.qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

        # Main doc collection
        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client, collection_name="agentic_rag_collection"
        )

        # (Optional) grounded memory — used to bias retrieval/intent only (never cited)
        self.memory_store = QdrantVectorStore(
            client=self.qdrant_client, collection_name="agentic_rag_memory"
        )
        self.memory_index = VectorStoreIndex.from_documents([], vector_store=self.memory_store)
        self.memory_retriever = VectorIndexRetriever(index=self.memory_index, similarity_top_k=3)
        self.use_grounded_memory = True
        self.memory_doc_count = 0
        self.max_memory_docs = 200
        self._last_memory_text = None

        # Indices & retrievers
        self.index = None
        self.retriever = None

        # Bias next retrieval toward last cited files (file names only)
        self._last_cited_files: set[str] = set()

        # Chat history (for intent only)
        self.chat_history = []

        # Neutral QA prompt (no labels, no citations)
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
                documents.extend(reader.load_data(path))

        # Token-based splitter (no NLTK dependency)
        node_parser = TokenTextSplitter(chunk_size=1024, chunk_overlap=120)

        self.index = VectorStoreIndex.from_documents(
            documents, vector_store=self.vector_store, node_parser=node_parser
        )
        # High-ish recall to cover lists/tables
        self.retriever = VectorIndexRetriever(index=self.index, similarity_top_k=TOP_K)

    # ---------- Helpers ----------
    @staticmethod
    def _format_chat_window(history, k=6):
        if not history:
            return "(empty)"
        window = history[-k:]
        lines = []
        for m in window:
            role = m.get("role", "user").capitalize()
            content = m.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    @staticmethod
    def _get_text(node):
        t = getattr(node, "text", None)
        if not t and hasattr(node, "get_content"):
            try:
                t = node.get_content()
            except Exception:
                t = None
        return (t or "").strip()

    def _context_from_nodes(self, nodes_docs) -> str:
        """Plain, readable context for the LLM (no labels)."""
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
        """Prefer chunks from the last-cited files; backfill if too few."""
        if not self._last_cited_files:
            return nodes
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

    def _embed(self, texts: List[str]) -> np.ndarray:
        embs = []
        for t in texts:
            try:
                embs.append(Settings.embed_model.get_text_embedding(t))
            except Exception:
                embs.append([0.0])
        maxd = max(len(e) for e in embs)
        embs = [np.pad(np.array(e, dtype="float32"), (0, maxd - len(e))) for e in embs]
        return np.vstack(embs)

    def _attach_citations(self, answer_text: str, nodes_docs: list) -> Tuple[str, List[Tuple[int, str]]]:
        """
        For each factual sentence in `answer_text`, find the best supporting doc chunk by embedding similarity.
        Append numeric [k] (1-based) where k is the first-appearance index of that chunk in this answer.
        Return (new_text, sources) where sources = [(k, 'file, page X'), ...] in order used.
        """
        sentences = self._sentences(answer_text)
        if not sentences:
            return "No answer found.", []

        # Prepare doc chunk texts & meta
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

        # Embeddings & similarity (dot product — good enough for E5)
        s_emb = self._embed(sentences)   # [S x d]
        d_emb = self._embed(doc_texts).T # [d x D]
        sims = s_emb @ d_emb             # [S x D]

        used_map, used_order = {}, []  # doc_idx -> label, and order of first use
        out_sentences = []

        for i, s in enumerate(sentences):
            # Ignore super short/non-factual sentences (e.g., "Here you go:")
            if len(re.findall(r"[A-Za-z0-9]", s)) < 6:
                continue

            j = int(np.argmax(sims[i]))
            score = float(sims[i, j])

            if score < SIM_THRESHOLD:
                # Not confidently supported by any retrieved chunk
                continue

            if j not in used_map:
                used_map[j] = len(used_map) + 1  # 1-based label for this doc chunk
                used_order.append(j)
            k = used_map[j]
            out_sentences.append(f"{s} [{k}]")

        if not out_sentences:
            return "No answer found.", []

        new_text = " ".join(out_sentences)
        sources = [(used_map[j], doc_meta[j]) for j in used_order]
        return new_text, sources

    # ---------- Grounded memory (optional, never cited) ----------
    def _save_memory(self, question: str, answer_text: str, sources: List[Tuple[int, str]]):
        """Persist grounded answer (with visible sources list) for future retrieval bias."""
        if not answer_text or answer_text.strip().lower() == "no answer found.":
            return
        if not sources:
            return
        srcs = "; ".join(f"[{k}] {v}" for k, v in sources)
        blob = f"Q: {question}\nA: {answer_text}\nSources: {srcs}".strip()
        doc = Document(text=blob, metadata={"source": "memory"})
        self.memory_index.insert(doc)
        self.memory_doc_count += 1
        self._last_memory_text = blob

        if self.memory_doc_count > self.max_memory_docs:
            self.memory_index = VectorStoreIndex.from_documents([], vector_store=self.memory_store)
            self.memory_retriever = VectorIndexRetriever(index=self.memory_index, similarity_top_k=3)
            self.memory_doc_count = 0
            self._last_memory_text = None

    # ---------- Chat query ----------
    def query(self, question: str):
        if self.retriever is None:
            return {"answer": ["No documents indexed yet."], "sources": [], "messages": self.chat_history}

        # 1) Retrieve from docs
        try:
            nodes_docs = self.retriever.retrieve(question) or []
        except Exception:
            nodes_docs = []
        nodes_docs = self._bias_nodes_to_last_files(nodes_docs)

        # Optional: retrieve from grounded memory to bias intent (never cited)
        if self.use_grounded_memory and (self.memory_doc_count > 0 or self._last_memory_text):
            try:
                _ = self.memory_retriever.retrieve(question)  # warms things up; not used directly
            except Exception:
                pass

        # If nothing retrieved → immediate no
        if not nodes_docs:
            self.chat_history.append({"role": "user", "content": question})
            self.chat_history.append({"role": "assistant", "content": "No answer found."})
            return {
                "answer": ["No answer found."],
                "sources": [],
                "messages": self.chat_history,
            }

        # 2) Build context and prompt
        context_str = self._context_from_nodes(nodes_docs)
        chat_window = self._format_chat_window(self.chat_history, k=6)

        final_prompt = self.qa_template.format(
            context_str=context_str,
            query_str=question,
            chat_window=chat_window,
        )

        # 3) LLM answer (no labels)
        chunks = []
        for part in Settings.llm.stream_complete(final_prompt):
            delta = getattr(part, "delta", None)
            text = delta if isinstance(delta, str) else getattr(part, "text", "")
            if text:
                chunks.append(text)
        raw_answer = "".join(chunks).strip()

        if not raw_answer or raw_answer.lower() == "no answer found.":
            full_text = "No answer found."
            ordered_sources = []
        else:
            # 4) Deterministic citations
            full_text, ordered_sources = self._attach_citations(raw_answer, nodes_docs)

        # 5) Update state & (optional) save grounded memory
        self.chat_history.append({"role": "user", "content": question})
        self.chat_history.append({"role": "assistant", "content": full_text})
        self._save_memory(question, full_text, ordered_sources)

        # Track last cited files to bias next retrieval
        cited_files = {src.split(",")[0].strip().lower() for _, src in ordered_sources}
        self._last_cited_files = cited_files

        # 6) Return streamed-style chunks + sources
        return {
            "answer": [full_text],          # we already built the final answer
            "sources": ordered_sources,     # [(1, "file, page X"), (2, "..."), ...]
            "messages": self.chat_history,
        }
