from dataclasses import dataclass
from io import BytesIO
from uuid import uuid4

import numpy as np
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    TOP_K,
)

@dataclass
class Chunk:
    chunk_id: str
    text: str
    page: int | None

class DocumentIndex:
    def __init__(self):
        self.chunks: list[Chunk] = []
        self.vectorizer = None
        self.matrix = None

    def _chunk_text(
        self,
        text: str,
        page: int | None,
    ) -> list[Chunk]:
        text = " ".join(text.split())

        if not text:
            return []

        chunks = []
        start = 0

        while start < len(text):
            end = min(
                len(text),
                start + CHUNK_SIZE,
            )

            chunks.append(
                Chunk(
                    chunk_id=str(uuid4()),
                    text=text[start:end],
                    page=page,
                )
            )

            if end == len(text):
                break

            start = max(
                end - CHUNK_OVERLAP,
                start + 1,
            )

        return chunks

    def ingest_bytes(
        self,
        data: bytes,
        filename: str,
    ) -> int:
        chunks = []

        if filename.lower().endswith(".pdf"):
            reader = PdfReader(BytesIO(data))

            for page_no, page in enumerate(
                reader.pages,
                start=1,
            ):
                text = page.extract_text() or ""
                chunks.extend(
                    self._chunk_text(
                        text,
                        page_no,
                    )
                )

        elif filename.lower().endswith(".txt"):
            text = data.decode(
                "utf-8",
                errors="ignore",
            )
            chunks.extend(
                self._chunk_text(
                    text,
                    None,
                )
            )

        else:
            raise ValueError(
                "Only PDF and TXT are supported."
            )

        if not chunks:
            raise ValueError(
                "No readable text was found."
            )

        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
        )
        self.matrix = self.vectorizer.fit_transform(
            [chunk.text for chunk in chunks]
        )

        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = TOP_K,
    ) -> list[dict]:
        if self.vectorizer is None:
            raise ValueError(
                "Upload and index a document first."
            )

        query_vector = self.vectorizer.transform(
            [query]
        )
        scores = cosine_similarity(
            query_vector,
            self.matrix,
        ).ravel()

        order = np.argsort(scores)[::-1][:top_k]

        return [
            {
                "chunk_id": self.chunks[i].chunk_id,
                "text": self.chunks[i].text,
                "page": self.chunks[i].page,
                "score": round(
                    float(scores[i]),
                    4,
                ),
            }
            for i in order
        ]
