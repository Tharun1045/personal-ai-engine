import hashlib
import tiktoken
from typing import List
from src.personal_ai_engine.domain.document import Document, DocumentChunk


def get_encoder(model_name: str = "gpt-4") -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def chunk_document(
    document: Document,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    model_name: str = "gpt-4",
) -> List[DocumentChunk]:
    """Split a Document into multiple DocumentChunks using token-based recursive splitting.

    Args:
        document: The document to chunk.
        chunk_size: Maximum number of tokens per chunk.
        chunk_overlap: Number of overlapping tokens between chunks.
        model_name: Model name for tiktoken encoding.

    Returns:
        List[DocumentChunk]: The generated document chunks.
    """
    if not document.content or not document.content.strip():
        return []

    encoder = get_encoder(model_name)
    tokens = encoder.encode(document.content)

    chunks = []
    start = 0
    chunk_idx = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_content = encoder.decode(chunk_tokens)

        # Generate deterministic chunk ID
        h = hashlib.sha256()
        h.update(f"{document.id}_{chunk_idx}_{chunk_content}".encode("utf-8"))
        chunk_id = h.hexdigest()

        metadata = {
            "parent_id": document.id,
            "title": document.metadata.title,
            "url": document.metadata.url,
            "chunk_index": chunk_idx,
        }

        chunk = DocumentChunk(
            id=chunk_id, parent_id=document.id, content=chunk_content, metadata=metadata
        )
        chunks.append(chunk)

        start += chunk_size - chunk_overlap
        chunk_idx += 1

        # Guard against infinite loop
        if chunk_size <= chunk_overlap:
            break

    return chunks
