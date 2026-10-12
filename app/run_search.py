from app.retriever import get_retriever


def main() -> None:
    retriever = get_retriever()

    query = input("Pergunta de busca: ").strip()

    results = retriever.search(query, top_k=4)

    for rank, result in enumerate(results, start=1):
        print(f"\n--- Result {rank} ---")
        print(f"Title: {result['title']}")
        print(f"Score: {result['score']:.4f}")
        print(result["text"])


if __name__ == "__main__":
    main()