# Phase 2: Data Pipelines - Handoff

## Summary
Completed the implementation of Phase 2 data ingestion pipelines based on the reference project, adapted for the `personal-ai-engine`.

## Key Implementations
- **Domain Mapping**: Introduced `Document` and `DocumentMetadata` domain models.
- **Notion Integration**: Implemented Notion API extraction using `requests` with robust error handling and pagination to fetch pages and child blocks from a configured Notion database.
- **Web Crawling**: Integrated `Crawl4AICrawler` to process URLs found in Notion documents concurrently.
- **Content Cleaning**: Implemented basic regex-based cleaning logic.
- **Quality Scoring**: Implemented `HeuristicQualityAgent` for URL-ratio based deterministic scoring, and adapted `QualityScoreAgent` to use the configured `AIGatewayFactory` with Ollama as the provider.
- **Idempotency**: Upgraded `MongoDBClient` to perform idempotent upserts using `pymongo.UpdateOne` and the document ID as key.
- **ZenML Pipelines**: Ported `ingestion_pipeline` and `etl_pipeline` to string together the extraction, crawling, scoring, and storage steps.
- **Testing**: Added integration and unit tests for Crawler logic, ETL steps, pipelines, and Notion clients. Verified local test execution passes.

## Notes & Constraints
- Only `ollama` is used as the LLM provider for the quality agent as requested.
- Provider switching can be done via environment variables (`AI_PROVIDER="ollama"`).
- Tests confirm the data pipelines can execute correctly without hitting external/paid LLM endpoints.

## Next Steps
- Setup the retrieval flows in Phase 3.
