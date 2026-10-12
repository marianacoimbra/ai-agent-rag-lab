from app.rag import answer_with_rag


def main() -> None:
    question = input("Faça uma pergunta: ").strip()

    if not question:
        print("A pergunta não pode estar vazia.")
        return

    result = answer_with_rag(question)

    print("\n=== ANSWER ===")
    print(result["answer"])

    print("\n=== SOURCES ===")

    for source in result["sources"]:
        print(
            f"\n[{source['reference']}] {source['title']} "
            f"(score={source['score']})"
        )
        print(source["text"][:500])


if __name__ == "__main__":
    main()