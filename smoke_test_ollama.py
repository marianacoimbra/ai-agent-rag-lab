from ollama import chat

response = chat(
    model="qwen3:4b",
    messages=[
        {
            "role": "user",
            "content": "Explain why does homophobia exists.",
        }
    ],
)

print(response.message.content)