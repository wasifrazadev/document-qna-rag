# RAG Chat API

Ask questions about your PDF documents through a REST API.

The project reads PDFs, stores their embeddings in a local Chroma database, and
answers questions using the most relevant chunks as context for a Mistral LLM.
Built with FastAPI, LangChain, HuggingFace embeddings and ChromaDB.

## How it works

```
 PDF ──► split into chunks ──► embeddings ──► Chroma DB      (scripts/ingest.py, run once)

 Question ──► retrieve top chunks (MMR) ──► prompt + Mistral ──► answer   (FastAPI, /ask)
```

## Project structure

```
rag-api/
├── app/
│   ├── main.py          # FastAPI app, startup, CORS
│   ├── routes.py        # /health and /ask endpoints
│   ├── rag.py           # retrieval + LLM logic
│   ├── prompts.py       # prompt template
│   ├── embeddings.py    # embedding model (shared with ingest)
│   ├── schemas.py       # request / response models
│   └── config.py        # settings loaded from .env
├── scripts/
│   └── ingest.py        # builds the vector DB from PDFs
├── docs/                # put your PDFs here
├── chroma-db/           # generated vector database (git-ignored)
├── .env.example
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10+
- A [Mistral API key](https://console.mistral.ai)

## Setup

**1. Create and activate a virtual environment**

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

The first install is large (PyTorch comes with `sentence-transformers`).

**3. Configure environment variables**

```bash
cp .env.example .env        # Windows: copy .env.example .env
```

Open `.env` and set `MISTRAL_API_KEY`.

**4. Add your PDF(s)**

Put them in the `docs/` folder, e.g. `docs/pp.pdf`.

**5. Create the embeddings (run once)**

```bash
python -m scripts.ingest
```

Other options:

```bash
python -m scripts.ingest docs/pp.pdf   # a single file
python -m scripts.ingest --reset       # wipe the DB and start fresh
```

The first run downloads the embedding model, so it may take a minute.

**6. Start the API**

```bash
uvicorn app.main:app --reload
```

The server runs at http://127.0.0.1:8000. Interactive docs are at
http://127.0.0.1:8000/docs.

## API

### `GET /health`

```json
{ "status": "ok" }
```

### `POST /ask`

Request:

```json
{ "question": "What is this document about?" }
```

Response:

```json
{
  "answer": "…",
  "sources": ["chunk text 1", "chunk text 2", "…"]
}
```

Example with curl:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?"}'
```

Example with Python:

```python
import requests

r = requests.post(
    "http://127.0.0.1:8000/ask",
    json={"question": "What is this document about?"},
)
print(r.json()["answer"])
```

## Configuration

All settings live in `.env` (see `.env.example`).

| Variable | Default | Description |
|---|---|---|
| `MISTRAL_API_KEY` | – | Required. Your Mistral API key |
| `MISTRAL_MODEL` | `mistral-tiny` | Mistral chat model |
| `EMBEDDING_MODEL` | `sentence-transformers/all-mpnet-base-v2` | HuggingFace embedding model |
| `CHROMA_DIR` | `chroma-db` | Where the vector DB is stored |
| `DOCS_DIR` | `docs` | Default folder scanned by ingest |
| `RETRIEVER_K` | `4` | Chunks sent to the LLM |
| `RETRIEVER_FETCH_K` | `10` | Candidates fetched before MMR re-ranking |
| `RETRIEVER_LAMBDA_MULT` | `0.5` | MMR diversity (0 = max diversity, 1 = max relevance) |
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins |

## Troubleshooting

**`Vector store not found`**: run `python -m scripts.ingest` before starting the API.

**`MISTRAL_API_KEY is not set`**: make sure `.env` exists in the project root and contains your key.

**Answers are bad or errors after changing the embedding model**: the DB must be built with the same model the API uses. Run `python -m scripts.ingest --reset`.

**Duplicate chunks in results**: you ingested the same PDF more than once. Use `--reset`.

**Added/changed a PDF**: stop the server, run `python -m scripts.ingest --reset`, then start it again.

## Production notes

- Set `ALLOWED_ORIGINS` to your real frontend URL(s) instead of `*`.
- Run without `--reload`, e.g. `uvicorn app.main:app --host 0.0.0.0 --port 8000`.
- Consider adding authentication (e.g. an API key header) before exposing the API publicly, since each request uses your Mistral credits.
