import json
import os

import pytest

from portfolio.config import ROOT
from portfolio.runtime import Services


@pytest.mark.embeddings
@pytest.mark.skipif(os.getenv("RUN_EMBEDDING_TESTS") != "1", reason="Opt-in actual HF model download/execution")
def test_real_huggingface_chroma_pipeline(settings, alice):
    configured = settings.model_copy(update={"embedding_backend": "huggingface", "embedding_local_only": True})
    service = Services(configured)
    try:
        session = service.store.new_session(alice, "Real embeddings")["id"]
        service.ingest(alice, session, "animals.txt", b"A dog is playing in the park. A puppy runs outdoors with its owner.")
        service.ingest(alice, session, "revenue.txt", b"The financial report contains quarterly revenue projections and tax liabilities.")
        hits = service.rag.retrieve(alice, "An animal enjoys outdoor play", strategy="vector", k=1)
        assert hits and hits[0]["source"] == "animals.txt"
        embedding = service.rag.vector_store(alice)._embedding_function
        dimension = len(embedding.embed_query("test dimension"))
        assert dimension == 384
        record = {"model": configured.embedding_model, "backend": "huggingface", "dimensions": dimension,
                  "query": "An animal enjoys outdoor play", "rank_1_source": hits[0]["source"],
                  "cosine_similarity": hits[0]["cosine_similarity"], "inference": "actual local model execution, no hosted LLM call"}
        (ROOT/"reports"/"embedding-validation.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    finally:
        service.close()
