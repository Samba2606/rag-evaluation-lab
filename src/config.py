import os

from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "llama-3.3-70b-versatile",
)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "4"))
