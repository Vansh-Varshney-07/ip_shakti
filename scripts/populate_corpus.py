"""Populate the persistent IP-SAKTI retrieval and graph stores from data/corpus."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ip_sakti.config.loader import get_settings
from ip_sakti.core.models import AuthorityTier, DocumentType, JurisdictionCode, IngestionJob
from ip_sakti.ingestion.pipeline import create_ingestion_pipeline
from ip_sakti.retrieval.retrieval_engine import create_retrieval_engine


TYPE_BY_DIRECTORY = {
    "acts": DocumentType.ACT,
    "rules": DocumentType.RULE,
    "treaties": DocumentType.TREATY,
    "protocols": DocumentType.PROTOCOL,
    "forms": DocumentType.FORMULARY,
    "guidelines": DocumentType.GUIDELINE,
}

TIER_BY_DIRECTORY = {
    "acts": AuthorityTier.TIER_1,
    "rules": AuthorityTier.TIER_2,
    "treaties": AuthorityTier.TIER_1,
    "protocols": AuthorityTier.TIER_1,
    "forms": AuthorityTier.TIER_5,
    "guidelines": AuthorityTier.TIER_5,
}


async def main() -> None:
    settings = get_settings()
    root = Path(settings.corpus_root or "data/corpus")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    engine = create_retrieval_engine()
    await engine.initialize()
    pipeline = create_ingestion_pipeline(retrieval_engine=engine)
    total_chunks = 0
    failures = []
    indexed_paths = {
        chunk.metadata.get("source_path")
        for chunk in getattr(engine.vector_store, "chunks", {}).values()
        if chunk.metadata.get("source_path")
    }

    try:
        for path in files:
            if str(path) in indexed_paths:
                print(f"{path.name}: already indexed")
                continue
            document_type = TYPE_BY_DIRECTORY.get(path.parent.name, DocumentType.ACT)
            if "wipo" in path.parts or "cbd_nagoya" in path.parts:
                document_type = DocumentType.PROTOCOL if "protocol" in path.stem.lower() or "nagoya" in path.stem.lower() else DocumentType.TREATY
            jurisdiction = (
                JurisdictionCode.INTERNATIONAL
                if "wipo" in path.parts or "cbd_nagoya" in path.parts
                else JurisdictionCode.INDIA
            )
            job = IngestionJob(
                source_path=str(path),
                title=path.name,
                document_type=document_type,
                jurisdiction=jurisdiction,
                authority_tier=TIER_BY_DIRECTORY.get(path.parent.name, AuthorityTier.TIER_2),
                metadata={"gazette_published": True},
            )
            result = await pipeline.ingest(job)
            chunk_count = len(result.chunks)
            total_chunks += chunk_count
            print(f"{path.name}: {result.status.value} ({chunk_count} chunks)")
            if result.error_message:
                failures.append({"path": str(path), "error": result.error_message})

        print(f"SUMMARY files={len(files)} chunks={total_chunks} failed={len(failures)}")
        if failures:
            for failure in failures:
                print(f"FAILED {failure['path']}: {failure['error']}")
        print(f"STATS {await engine.get_stats()}")
    finally:
        await engine.close()


if __name__ == "__main__":
    asyncio.run(main())
