import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
INDEX_DIR = DATA_DIR / "index"

DATASET_NAME = os.getenv("HF_DATASET_NAME", "rajpurkar/squad")
SAMPLE_LIMIT = int(os.getenv("SAMPLE_LIMIT", "500"))

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)