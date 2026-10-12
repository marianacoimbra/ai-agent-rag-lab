import json
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL, INDEX_DIR


class LocalRetriever:
    def __init__(self) -> None:
        embeddings_path = INDEX_DIR / "embeddings.npy"
        chunks_path = INDEX_DIR / "chunks.json"

        if not embeddings_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(
                "Index not found. Run: python -m app.ingest"
            )

        self.embeddings = np.load(embeddings_path).astype(np.float32)

        with chunks_path.open("r", encoding="utf-8") as file:
            self.chunks = json.load(file)

        if len(self.chunks) != len(self.embeddings):
            raise ValueError(
                "Embedding count does not match chunk count."
            )

        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def search(
        self,
        query: str,
        top_k: int = 4,
    ) -> list[dict]:
        if not query.strip():
            return []

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        # Prevent unnecessarily large responses.
        top_k = min(top_k, 10)

        query_vector = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0].astype(np.float32)

        # Both vectors and query are normalized:
        # dot product is equivalent to cosine similarity.
        scores = self.embeddings @ query_vector

        indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for index in indices:
            result = dict(self.chunks[int(index)])
            result["score"] = float(scores[int(index)])
            results.append(result)

        return results


@lru_cache(maxsize=1)
def get_retriever() -> LocalRetriever:
    """Reuse the loaded vector index and embedding model."""
    return LocalRetriever()