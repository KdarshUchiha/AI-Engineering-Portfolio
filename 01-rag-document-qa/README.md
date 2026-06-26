# Project 1 — Smart Document Q&A (RAG)

Ask questions about any text document and get accurate, **cited** answers from
Claude. Built to teach the single most in-demand AI engineering pattern:
**Retrieval-Augmented Generation (RAG)**.

## Why this project matters for an FDE portfolio

RAG is the #1 thing enterprises hire AI engineers to build right now: "let our
employees/customers ask questions over our internal docs." If you can build,
explain, and demo this, you can walk into a forward-deployed interview and talk
about it like a consultant. This repo is deliberately written so you understand
**every line**, not just call a library.

## How it works — the 4 steps of RAG

```
Document ──▶ 1. CHUNK ──▶ 2. EMBED ──▶ [vector index]
                                            │
Question ───────────────▶ 2. EMBED ──▶ 3. RETRIEVE (cosine similarity)
                                            │
                                   top-k relevant chunks
                                            │
                                            ▼
                                   4. GENERATE (Claude) ──▶ cited answer
```

1. **Chunk** — split the document into small overlapping passages.
2. **Embed** — turn each passage (and the question) into a vector of numbers
   that captures meaning. We use transparent TF-IDF math you can read.
3. **Retrieve** — find the passages whose vectors are closest to the question.
4. **Generate** — give Claude *only* those passages and ask it to answer using
   them, with citations.

## Files

| File                  | What it is                                              |
|-----------------------|---------------------------------------------------------|
| `rag.py`              | The whole engine, heavily commented. Read top to bottom.|
| `sample_document.txt` | A document about the FDE role, to query against.        |
| `README.md`           | This file.                                              |

## Run it

### 1. Install the one dependency
```bash
pip install anthropic
```
(Retrieval uses only the Python standard library — no numpy, no ML packages.)

### 2. Get a Claude API key
- Go to https://console.anthropic.com → **API Keys** → create one.
- It looks like `sk-ant-...`. Add a few dollars of credit.

### 3. Set the key as an environment variable
```bash
export ANTHROPIC_API_KEY="sk-ant-your-key-here"
```
(Never paste the key into the code — env vars keep secrets out of git.)

### 4. Run
```bash
python rag.py
```
You'll see it index the document, then answer three questions — including one
that's **not** in the document, to prove it says "I don't know" instead of
making something up.

## Try your own document

Replace `sample_document.txt` with any `.txt` file (your resume, a product spec,
meeting notes), rerun, and ask about it. Or import the class:

```python
from rag import DocumentQA
qa = DocumentQA(open("my_doc.txt").read())
print(qa.ask("What are the key risks mentioned?"))
```

## Cost note

Each question is one Claude API call. With `claude-opus-4-8` that's a fraction
of a cent for a short Q&A. To cut cost ~5x for high volume, change the `model`
in `generate_answer()` to `claude-sonnet-4-6`, or `claude-haiku-4-5` for the
cheapest/fastest option.

## What to learn from this before moving on

- **Chunking + overlap** — why and how documents are split.
- **Embeddings & vectors** — text as numbers; we used TF-IDF, production uses
  neural embedding models (a one-function swap, marked in `rag.py`).
- **Cosine similarity** — how "closeness" of meaning is measured.
- **Grounded generation** — giving the model context so it doesn't hallucinate.
- **The Anthropic SDK** — `client.messages.create(...)`, system prompts,
  reading `response.content`.

## Upgrade ideas (great for your LinkedIn post / interview talking points)

- Swap TF-IDF for real embeddings (Voyage/sentence-transformers) for smarter
  semantic search.
- Add a Streamlit or FastAPI web UI so it's clickable, not just a terminal.
- Support PDFs (Claude can read PDFs directly via the Files API).
- Stream the answer token-by-token for a ChatGPT-style feel.
