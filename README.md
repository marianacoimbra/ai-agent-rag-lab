# AI Agent with RAG — Local Lab

## 1. Project Goal
The goal of this lab is to build an AI agent from scratch with Python with the aim of understanding each component before introducing frameworks such as PydanticAI.

After implementing the RAG pipeline with Hugging Face data, local embeddings, semantic search, and response generation, then I gradually turn that search into a tool that an LLM can decide to use, and finally compare its implementation with PydanticAI.

## 2. Learning Objectives
I pursue to develop my AI Engineering skills and knowledge, improving my understanding on each stage.

## 3. Architecture
[Architecture Diagram](docs/architecture.md)

## 4. Technology Stack

Python 3.11, Poetry, Pydantic, Sentence Transformers, Hugging Face Datasets, FAISS, LangChain, Ollama (local LLM), and more.

## 5. Prerequisites
python 3.11, Poetry, and Ollama installed on your machine.

## 6. Environment Setup
pip install -r requirements.txt

## 7. Dataset and Ingestion
run `python -m app.ingest` to download the dataset and generate the embeddings.

## 8. Embeddings and Retrieval
local embeddings are generated using Sentence Transformers, and FAISS is used for similarity search. The retrieval process involves embedding the query and searching for relevant chunks in the indexed dataset.

## 9. RAG Pipeline
located in `app.rag`, the RAG pipeline combines the retrieval of relevant chunks with the generation of responses using a local LLM (Ollama). The retrieved chunks are passed to the LLM to generate a coherent answer with sources.

## 10. Manual Agent and Tool Calling
This was part of the learning process, where I implemented a manual agent that can call the search tool to retrieve relevant information. The manual agent interacts with the LLM and decides when to use the search tool based on the query.

## 11. PydanticAI Implementation

After understanding the manual agent and tool calling, I implemented a similar agent using PydanticAI. This agent leverages the capabilities of PydanticAI to manage the interaction with the LLM and the search tool more efficiently, allowing for a more streamlined and effective response generation process.

## 12. Running the Tests

To run the tests, execute the following command in your terminal:

```bash
pytest -q
```

## 13. References
- [Pydantic Core Concepts: Agent](https://pydantic.dev/docs/ai/core-concepts/agent/)
- [LangChain Documentation](https://docs.langchain.com/oss/python/learn)
- [Hugging Face Datasets](https://huggingface.co/datasets)
- [Ollama Library](https://ollama.com/docs)