# Project Context

## Purpose
The `personal-ai-engine` is an AI-powered offline and online system to ingest, embed, and retrieve data, as well as serve a UI.

## Architectural Reference
This project follows the architecture and learning path of the Decoding AI Second Brain AI Assistant course:
https://github.com/decodingai-magazine/second-brain-ai-assistant-course

## Offline vs Online Architecture
- **Offline application:** notion ingestion, crawling child URLs, text cleaning, quality scoring, document chunking, batch embedding, dataset preparation, fine-tuning and retrieval-index creation.
- **Online application:** hybrid search retrieval (RRF), inference, agent tools, Gradio chatbot UI, tracing and evaluations.

## Provider-Independent Design
The core logic remains decoupled from specific AI provider SDKs. We use Gateway factory and interfaces (`ChatGenerator`, `TextEmbedder`) that are implemented by Adapters.
- The active AI provider is strictly controlled by `AI_PROVIDER` and `EMBEDDING_PROVIDER` environment variables (`ollama`, `openrouter`, `gemini`).
- **No Silent Fallback:** The system will never silently fall back from one provider to another. 
- **Ollama Models:** Local defaults are explicitly set to `qwen3:8b` for chat and `nomic-embed-text:latest` for embeddings.
- Cloud providers (Gemini, OpenRouter) remain mocked in tests until explicitly authorized.

## Development Commands
- Dependency management: `uv`
- Synchronize environment: `uv sync`
- Formatting & Linting: `uv run ruff format .` and `uv run ruff check .`
- Type checking: `uv run mypy .`
- Tests: `uv run pytest -m "not integration"` (Unit tests) and `uv run pytest -m integration` (Smoke tests)
- Infrastructure: `docker compose -f infra/docker-compose.yml up -d`

## Implementation Handoffs
- Phase 1: Foundational structure, configuration handling (`pydantic-settings`), Docker Compose (MongoDB + ZenML), provider-independent AI gateway.
- Phase 2: Ingestion & ETL pipelines (Notion API query, block extraction, crawling child URLs via Crawl4AI, cleaning, and quality scoring).
- Full Project Adaptation: Implemented separate document chunking, batch embedding, local and Atlas MongoDB Vector Search backend, hybrid RRF search retriever, agent retrieval tool, Gradio Chatbot UI, Opik tracing wrapper, LLM-based RAG evaluation system, and Alpaca QA dataset generation + trainer skip.
