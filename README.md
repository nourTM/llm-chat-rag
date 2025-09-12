📚 Agentic RAG Batch QA Engine

This project is a Retrieval-Augmented Generation (RAG) pipeline for answering questions over custom PDF documents in batch mode.

It uses LlamaIndex with Qdrant as a vector store, and supports multiple LLM backends:

🐪 Ollama (local models like LLaMA, Qwen, Mistral, etc.)

💻 LM Studio (OpenAI-compatible local server)

☁️ OpenAI / Together / other OpenAI-compatible APIs

✨ Features

Batch Q&A: Provide an Excel/CSV file with a column of questions → get answers + citations back in a new Excel file.

Citations: Each factual sentence is automatically grounded to a retrieved chunk, with deterministic [1], [2] references.

Multilingual support: Uses multilingual E5 embeddings for retrieval, works with Arabic, English, and more.

Vector database: Stores documents in Qdrant for fast semantic retrieval.

Pluggable LLMs: Run locally with Ollama, LM Studio, or remotely via OpenAI.

Memory biasing (optional): Reuses past Q&A results to guide intent recognition.

🚀 How It Works

Index PDFs

Place your PDFs in a folder.

The system splits them into semantic chunks and stores embeddings in Qdrant.

Run batch questions

Provide an Excel/CSV with a question column.

Each question is answered using only retrieved context.

Answers are grounded with citations (file, page).

Output

An Excel file with:

question

answer

sources

🛠️ Setup
git clone https://github.com/<your-username>/llm-chat-rag.git
cd llm-chat-rag
python -m venv .venv
.venv\Scripts\activate   # Windows
source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt


Create a .env file (examples for Ollama / LM Studio / OpenAI included in docs).

📊 Usage
python rag_batch_cli.py --pdf-dir ./docs --input questions.xlsx --output answers.xlsx

⚡ Example

Input Excel (questions.xlsx):

question
What are the principles of personal data processing?
Could you give me contact info of Saudi Economy Watch?

Output Excel (answers.xlsx):

question	answer	sources
What are the principles of personal data processing?	Use correct tools (SPSS, R, SAS) [1] and modern techniques like machine learning [2]	[1] law.pdf, page 4; [2] guide.pdf, page 12
🔧 Future Improvements

OCR integration for scanned PDFs

Streamlit web UI for uploading files & asking questions interactively

Smarter memory for conversational QA
