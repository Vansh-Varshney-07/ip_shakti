import os

import pytest


os.environ.setdefault("IP_SAKTI_TEST_MODE", "true")


@pytest.mark.asyncio
async def test_ingestion_indexes_and_retrieves(tmp_path):
    from ip_sakti.core.models import DocumentChunk
    from ip_sakti.retrieval.retrieval_engine import RetrievalConfig, RetrievalEngine

    engine = RetrievalEngine(RetrievalConfig(enable_caching=False))
    await engine.initialize()
    try:
        chunk = DocumentChunk(
            id="smoke-chunk",
            document_id="smoke-document",
            content="Section 3 of the Patents Act concerns patentability.",
            metadata={"jurisdiction": "INDIA", "document_type": "act", "source_authority_tier": "TIER_1"},
        )
        await engine.index_chunks([chunk])
        response = await engine.search(__import__("ip_sakti.retrieval.retrieval_engine", fromlist=["SearchRequest"]).SearchRequest(query="patentability", top_k=3))
        assert response.results
        assert response.results[0].chunk.id == "smoke-chunk"
    finally:
        await engine.close()


def test_corpus_validator_flags_empty_files(tmp_path):
    from ip_sakti.ingestion.corpus_validation import validate_corpus

    (tmp_path / "empty.txt").write_bytes(b"")
    report = validate_corpus(tmp_path)
    assert not report["valid"]
    assert report["issues"][0]["issue"] == "zero_byte_file"
