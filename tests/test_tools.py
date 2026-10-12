import json

from app import tools


def test_search_knowledge_base_returns_json(monkeypatch):
    class FakeRetriever:
        def search(self, query: str, top_k: int = 3):
            assert query == "What is RAG?"
            assert top_k == 2

            return [
                {
                    "title": "RAG",
                    "source": "test",
                    "score": 0.91,
                    "text": "RAG combines retrieval and generation.",
                }
            ]

    monkeypatch.setattr(
        tools,
        "get_retriever",
        lambda: FakeRetriever(),
    )

    output = tools.search_knowledge_base(
        query="What is RAG?",
        top_k=2,
    )

    payload = json.loads(output)

    assert len(payload) == 1
    assert payload[0]["title"] == "RAG"
    assert payload[0]["score"] == 0.91