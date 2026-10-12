def chunk_text(
    text: str,
    chunk_size: int = 900,
    overlap: int = 120,
) -> list[str]:
    """
    Split text into overlapping character-based chunks.

    chunk_size: maximum characters per chunk.
    overlap: number of characters repeated between adjacent chunks.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    text = text.strip()

    if not text:
        return []

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks


def chunk_documents(
    documents: list[dict[str, str]],
    chunk_size: int = 900,
    overlap: int = 120,
) -> list[dict[str, str]]:
    """Split documents while preserving their metadata."""

    chunks: list[dict[str, str]] = []

    for document in documents:
        parts = chunk_text(
            document["text"],
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for part_index, part in enumerate(parts):
            chunks.append(
                {
                    "id": f"{document['id']}-chunk-{part_index:03d}",
                    "title": document["title"],
                    "source": document["source"],
                    "text": part,
                }
            )

    return chunks