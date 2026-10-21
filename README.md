# LangChain PDF RAG

A runnable Python demo by **bhavindobariya7777**: scan a local PDF folder, split text into chunks, create OpenAI embeddings, persist vectors in local **ChromaDB**, and answer questions with file and page references.

## How it works

1. Recursively scan `documents/` for PDFs (including subfolders and uppercase extensions).
2. Extract text with pypdf, preserving one-based page numbers.
3. Split pages with LangChain's recursive text splitter (1,000 characters, 150 overlap).
4. Embed chunks using `text-embedding-3-small` and save them to disk in ChromaDB.
5. Embed the question, retrieve the top four chunks, and pass them to the chat model.
6. Show the answer, numbered citations, and supporting excerpts.

The daily worker scans immediately on startup and then at **02:00 Asia/Kolkata** by default. SHA-256 fingerprints skip unchanged files. Changed files replace their previous chunks; removed files are removed from the index. Failed PDF replacements preserve the previous complete version. A file lock coordinates local UI and worker operations. Interrupted batches are retried on the next scan.

## Quick start (Python 3.11+)

Run commands from the repository root so relative document and data paths match across processes.

```bash
git clone https://github.com/bhavindobariya7777/langchain-pdf-rag.git
cd langchain-pdf-rag
python3 -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e .
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env`, then copy your PDFs into `documents/`.

```bash
pdf-rag ingest
streamlit run app.py
```

Open the local URL printed by Streamlit. Click **Scan PDFs now** after adding documents, or use the CLI:

```bash
pdf-rag ask "What is the leave policy?"
```

Each question is standalone. The UI keeps a display history, but previous messages are not sent to the model.

## Run the daily scan

In a second terminal, activate the same environment and run:

```bash
pdf-rag worker
```

**The computer/server and worker must remain running.** Pushing code to GitHub does not run the worker or scan your computer's folder. For continuous use, supervise this command with your server's process manager. The worker catches up with a scan on restart.

Alternatively schedule `pdf-rag ingest` using cron on your own machine. Use absolute paths, change to the repository directory, and configure the cron host timezone explicitly. Do not run an additional worker when using cron.

## Configuration

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | required | OpenAI API credential |
| `PDF_RAG_DOCUMENTS_DIR` | `documents` | Folder scanned recursively |
| `PDF_RAG_DATA_DIR` | `data` | Persistent vectors and process lock |
| `PDF_RAG_EMBEDDING_MODEL` | `text-embedding-3-small` | Document and query embeddings |
| `PDF_RAG_CHAT_MODEL` | `gpt-4o-mini` | Answer generation |
| `PDF_RAG_CHUNK_SIZE` | `1000` | Characters per chunk |
| `PDF_RAG_CHUNK_OVERLAP` | `150` | Overlapping characters |
| `PDF_RAG_TOP_K` | `4` | Retrieved chunks |
| `PDF_RAG_SCAN_HOUR` | `2` | Daily hour, 0–23 |
| `PDF_RAG_SCAN_MINUTE` | `0` | Daily minute, 0–59 |
| `PDF_RAG_TIMEZONE` | `Asia/Kolkata` | IANA timezone |

Embedding/chunk configuration changes select a separate collection, requiring a new ingestion run. Old collections remain on disk; after stopping all processes you can delete `data/` and re-ingest to fully rebuild. Always restart the UI and worker after changing configuration.

## Cost, privacy, and limitations

- Local ChromaDB requires no hosted database account or database subscription. OpenAI API calls are billed separately; a ChatGPT subscription does not supply API credit.
- PDF text is sent to OpenAI for embeddings, and retrieved excerpts are sent for answers. Use documents you are authorized to process.
- `.env`, PDFs, and `data/` are gitignored. Never commit API keys or confidential documents.
- Image-only/scanned PDFs need OCR first. Encrypted or malformed PDFs are reported as ingestion errors. Other PDFs continue processing.
- Citations identify retrieved excerpts; generated answers can still be wrong. Review the supporting text.
- This is a local demo without authentication. Do not expose the Streamlit server publicly without adding authentication and deployment protections.
- Large PDFs are loaded in memory. This is intended for modest local document sets, not distributed ingestion.
- Avoid modifying a PDF while ingestion is running. Failed interrupted writes are retried, but Chroma operations are not a multi-file transaction.

## Development and verification

```bash
pip install -e ".[dev]"
ruff check .
ruff format --check .
pytest -q
```

Tests use generated PDFs, a real persistent Chroma database, deterministic fake embeddings, and a fake chat model. They cover ingestion, unchanged-file skipping, replacement, failed replacement, deletion, citations, partial-write repair, and settings validation without an API key.

A live OpenAI end-to-end run requires your own API key and billing. The GitHub Actions workflow runs formatting, linting, and offline tests on pushes and pull requests.

## Project layout

```text
app.py                   Streamlit question-answer UI
src/pdf_rag/config.py    Validated environment settings
src/pdf_rag/service.py   PDF ingestion, persistent vectors, retrieval, answers
src/pdf_rag/cli.py       One-off ingestion, daily worker, CLI questions
tests/test_rag.py        Offline integration tests
documents/              Your local PDFs (gitignored)
data/                   Generated local index (gitignored)
```

References: [LangChain Chroma integration](https://docs.langchain.com/oss/python/integrations/vectorstores/chroma) · [OpenAI embeddings integration](https://docs.langchain.com/oss/python/integrations/text_embedding/openai)
