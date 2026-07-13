from src.personal_ai_engine.domain.document import Document, DocumentMetadata
from src.offline.etl.chunking import chunk_document


def test_chunk_document_basic():
    meta = DocumentMetadata(
        id="doc1", url="http://test.com", title="Test Doc", properties={}
    )
    # Create document with 10 words (approx 12-15 tokens)
    doc = Document(
        id="doc1",
        metadata=meta,
        content="This is a test document with exactly ten words in it.",
    )

    # Chunk size larger than doc token count -> should return 1 chunk
    chunks = chunk_document(doc, chunk_size=50, chunk_overlap=5)
    assert len(chunks) == 1
    assert chunks[0].parent_id == "doc1"
    assert chunks[0].content == "This is a test document with exactly ten words in it."
    assert chunks[0].metadata["title"] == "Test Doc"


def test_chunk_document_splitting():
    meta = DocumentMetadata(
        id="doc2", url="http://test.com/2", title="Split Doc", properties={}
    )
    doc = Document(
        id="doc2",
        metadata=meta,
        content="one two three four five six seven eight nine ten",
    )

    # Small chunk size should split into multiple chunks
    chunks = chunk_document(doc, chunk_size=5, chunk_overlap=1)
    assert len(chunks) > 1
    # Check deterministic ID format
    assert len(chunks[0].id) == 64  # sha256 hex length


def test_chunk_document_empty():
    meta = DocumentMetadata(
        id="doc3", url="http://test.com/3", title="Empty Doc", properties={}
    )
    doc = Document(id="doc3", metadata=meta, content="")
    chunks = chunk_document(doc)
    assert len(chunks) == 0
