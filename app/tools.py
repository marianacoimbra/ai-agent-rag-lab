import json

from app.retriever import get_retriever


def search_knowledge_base(
    query: str,
    top_k: int = 3,
) -> str:
    """
    Search the local knowledge base for relevant text passages.

    Args:
        query: The information to look for.
        top_k: Maximum number of passages to retrieve.

    Returns:
        JSON containing the matched passages and their metadata.
    """

    results = get_retriever().search(
        query=query,
        top_k=top_k,
    )

    payload = [
        {
            "title": item["title"],
            "source": item["source"],
            "score": round(item["score"], 4),
            "text": item["text"],
        }
        for item in results
    ]

    return json.dumps(
        payload,
        ensure_ascii=False,
    )