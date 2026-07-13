# Project Context

## Purpose
The `personal-ai-engine` is an AI-powered offline and online system to ingest, embed, and retrieve data, as well as serve a UI.

## Architectural Reference
This project follows the architecture and learning path of the Decoding AI Second Brain AI Assistant course:
https://github.com/decodingai-magazine/second-brain-ai-assistant-course

## Offline vs Online Architecture
- **Offline application:** ingestion, ETL, dataset preparation, fine-tuning and retrieval-index creation.
- **Online application:** retrieval, inference, tools, UI, tracing and evaluations.

## Planned Phases
The project consists of 5 planned phases, starting with foundational structure and expanding into advanced Retrieval-Augmented Generation (RAG) and Fine-Tuning.

## Provider-Independent Design
The core logic remains decoupled from specific AI provider SDKs. We use Gateway interfaces (`ChatGenerator`, `TextEmbedder`) that are implemented by Adapters.
- The active AI provider is strictly controlled by the `AI_PROVIDER` environment variable (`ollama`, `openrouter`, `gemini`).
- **No Cloud Fallback:** The system will never silently fall back from one provider to another. 
- **Ollama Models:** Local defaults are explicitly set to `qwen3:8b` for chat and `nomic-embed-text:latest` for embeddings.
- Cloud providers (Gemini, OpenRouter) remain mocked in tests until explicitly authorized.

## Development Commands
- Dependency management: `uv`
- Formatting & Linting: `uv run ruff format .` and `uv run ruff check .`
- Type checking: `uv run mypy .`
- Tests: `uv run pytest -m "not integration"` (Unit tests) and `uv run pytest -m integration` (Smoke tests)
- Infrastructure: `docker compose -f infra/docker-compose.yml up -d`

## Phase 1 Implementation
Implemented the foundational project structure, configuration handling (`pydantic-settings`), Docker Compose for MongoDB and ZenML, provider-independent AI gateway, unit and mocked testing.

## Deferred to Later Phases
- Notion ingestion and ETL
- RAG, Embeddings Indexing, Vector Search
- Fine-Tuning, Agents, Tools, and UI.
