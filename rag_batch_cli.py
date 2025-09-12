# rag_batch_cli.py
import os
import glob
import argparse
import io
import pandas as pd

from rag_engine import RAGEngine


class _FileUploadShim:
    """
    Minimal in-memory file-like object to satisfy RAGEngine.load_documents(uploaded_files):
      - has .name
      - has .getbuffer() -> bytes-like
    """
    def __init__(self, path: str):
        self.name = os.path.basename(path)
        with open(path, "rb") as f:
            self._buf = f.read()

    def getbuffer(self):
        # Return a bytes-like object (memoryview works fine)
        return memoryview(self._buf)


def _collect_pdf_uploads(pdf_dir: str):
    pdf_paths = sorted(glob.glob(os.path.join(pdf_dir, "**", "*.pdf"), recursive=True))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found under: {pdf_dir}")
    return [_FileUploadShim(p) for p in pdf_paths]


def _load_questions_table(path: str, question_col: str):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif ext == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError("Input must be .xlsx/.xls or .csv")

    if question_col not in df.columns:
        # fallbacks
        for alt in ["Question", "questions", "query", "Query"]:
            if alt in df.columns:
                question_col = alt
                break
        else:
            raise KeyError(
                f"Could not find a '{question_col}' column in {path}. "
                "Use --question-col to specify a different column name."
            )

    # Normalize to string questions
    df[question_col] = df[question_col].fillna("").astype(str)
    return df, question_col


def main():
    ap = argparse.ArgumentParser(
        description="Batch Q&A over PDFs using RAGEngine (no edits to engine required)."
    )
    ap.add_argument("--pdf-dir", required=True, help="Directory containing PDFs (recursively indexed)")
    ap.add_argument("--input", required=True, help="Excel/CSV of questions (default column 'question')")
    ap.add_argument("--output", default="answers.xlsx", help="Output Excel path (default: answers.xlsx)")
    ap.add_argument("--question-col", default="Questions", help="Column name holding questions")
    args = ap.parse_args()

    # 1) Build upload shims from PDFs and index
    uploads = _collect_pdf_uploads(args.pdf_dir)
    eng = RAGEngine()
    eng.load_documents(uploads)

    # 2) Load question table
    df, qcol = _load_questions_table(args.input, args.question_col)

    # 3) Ask questions
    out_rows = []
    for q in df[qcol].tolist():
        res = eng.query(q)
        answer_text = (res.get("answer") or [""])[0]
        src_list = res.get("sources") or []
        sources_str = "; ".join(f"[{k}] {meta}" for k, meta in src_list)

        out_rows.append({
            qcol: q,
            "answer": answer_text,
            "sources": sources_str,
        })

    out_df = pd.DataFrame(out_rows, columns=[qcol, "answer", "sources"])
    out_df.to_excel(args.output, index=False)
    print(f"✅ Done. Wrote: {args.output}")


if __name__ == "__main__":
    main()
