import os
import time

import streamlit as st

def load_streamlit_secrets():
    try:
        if "GROQ_API_KEY" in st.secrets:
            os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

        if "MODEL_NAME" in st.secrets:
            os.environ["MODEL_NAME"] = st.secrets["MODEL_NAME"]
    except Exception:
        pass

load_streamlit_secrets()

from src.document_store import DocumentIndex
from src.llm import (
    answer_question,
    groundedness_judge,
)
from src.metrics import lexical_overlap

st.set_page_config(
    page_title="RAG Evaluation Lab",
    page_icon="📚",
    layout="wide",
)

st.title("RAG Evaluation Lab")
st.caption(
    "Upload a document, ask questions, inspect retrieval, and evaluate answer quality."
)

if "index" not in st.session_state:
    st.session_state.index = DocumentIndex()

uploaded = st.file_uploader(
    "Upload a PDF or TXT file",
    type=["pdf", "txt"],
)

if uploaded is not None:
    if st.button("Build RAG index"):
        try:
            count = st.session_state.index.ingest_bytes(
                uploaded.getvalue(),
                uploaded.name,
            )
            st.success(
                f"Indexed {count} chunks."
            )
        except Exception as exc:
            st.error(str(exc))

question = st.text_input(
    "Ask a question from the document"
)

if st.button("Ask and evaluate"):
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        try:
            started = time.perf_counter()

            contexts = st.session_state.index.search(
                question
            )

            answer = answer_question(
                question,
                contexts,
            )

            latency_ms = (
                time.perf_counter() - started
            ) * 1000

            lexical = lexical_overlap(
                answer,
                contexts,
            )

            judge = groundedness_judge(
                question,
                answer,
                contexts,
            )

            st.subheader("Answer")
            st.write(answer)

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Latency",
                f"{latency_ms:.0f} ms",
            )
            col2.metric(
                "Lexical grounding",
                f"{lexical:.2f}",
            )
            col3.metric(
                "LLM groundedness",
                f"{judge['score']}/100",
            )

            st.caption(
                f"Judge reason: {judge['reason']}"
            )

            st.subheader("Retrieved chunks")

            for i, context in enumerate(
                contexts,
                start=1,
            ):
                with st.expander(
                    (
                        f"Source {i} | "
                        f"page={context['page']} | "
                        f"score={context['score']}"
                    )
                ):
                    st.write(context["text"])

        except Exception as exc:
            st.error(str(exc))
