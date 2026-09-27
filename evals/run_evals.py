from pathlib import Path

from src.document_store import DocumentIndex

ROOT = Path(__file__).resolve().parent
SAMPLE = ROOT / "sample_document.txt"

index = DocumentIndex()
index.ingest_bytes(
    SAMPLE.read_bytes(),
    SAMPLE.name,
)

tests = [
    (
        "What does the handbook say about refunds?",
        "refund",
    ),
    (
        "What does it say about customer privacy?",
        "private",
    ),
]

passed = 0

for question, expected_word in tests:
    hits = index.search(
        question,
        top_k=3,
    )

    retrieved_text = " ".join(
        hit["text"].lower()
        for hit in hits
    )

    ok = expected_word.lower() in retrieved_text

    print(
        "PASS" if ok else "FAIL",
        "|",
        question,
    )

    passed += int(ok)

print(
    f"\nRetrieval pass rate: "
    f"{passed / len(tests):.0%}"
)
