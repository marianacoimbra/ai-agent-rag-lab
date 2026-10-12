import json

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from app.chunking import chunk_documents
from app.config import (
    DATASET_NAME,
    EMBEDDING_MODEL,
    INDEX_DIR,
    SAMPLE_LIMIT,
)


def load_documents() -> list[dict[str, str]]:
    dataset = load_dataset(
        DATASET_NAME,
        split=f"train[:{SAMPLE_LIMIT}]",
    )

    documents: list[dict[str, str]] = []
    seen_contexts: set[str] = set()

    for row in dataset:
        text = str(row.get("context", "")).strip()
        title = str(row.get("title", "Untitled")).strip()

        if not text or text in seen_contexts:
            continue

        seen_contexts.add(text)

        documents.append(
            {
                "id": f"doc-{len(documents):04d}",
                "title": title,
                "source": DATASET_NAME,
                "text": text,
            }
        )

    return documents


def build_index() -> None:
    documents = load_documents()
    chunks = chunk_documents(documents)

    if not chunks:
        raise RuntimeError("No chunks were generated from the dataset.")

    print(f"Documents: {len(documents)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Loading embedding model: {EMBEDDING_MODEL}")

    model = SentenceTransformer(EMBEDDING_MODEL)

    vectors = model.encode(
        [chunk["text"] for chunk in chunks],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    np.save(
        INDEX_DIR / "embeddings.npy",
        vectors.astype(np.float32),
    )

    with (INDEX_DIR / "chunks.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Index saved to: {INDEX_DIR}")
    print(f"Vector matrix shape: {vectors.shape}")


if __name__ == "__main__":
    build_index()