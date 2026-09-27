# RAG Evaluation Lab

A Streamlit-based AI Engineer portfolio project that demonstrates both RAG and evaluation.

## Features

- Upload PDF or TXT
- Chunk document text
- Build retrieval index
- Ask grounded questions
- Inspect retrieved chunks
- Measure retrieval scores
- Measure latency
- Measure lexical grounding
- Optional LLM groundedness judge

## Local setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this folder to GitHub.
2. Create a Streamlit app.
3. Main file: `app.py`
4. Add Secrets:

```toml
GROQ_API_KEY="your_key_here"
MODEL_NAME="llama-3.3-70b-versatile"
```

5. Deploy.

## What to explain in interviews

- Why retrieval must be evaluated separately from generation
- What top-k retrieval means
- How chunk size and overlap affect recall
- Why answer groundedness matters
- Why latency is an engineering metric
- How this can later be upgraded with:
  - dense embeddings
  - Qdrant
  - BM25
  - reranking
  - RAGAS
  - LangSmith
