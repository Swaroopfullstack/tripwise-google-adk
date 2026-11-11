"""Reproducible public-data experiments; metrics describe these fixtures only."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
from time import perf_counter

from .config import ROOT, Principal, Settings
from .data import read_csv
from .ml import train_models
from .public_data import public_file
from .runtime import Services


def evaluate():
    settings = Settings()
    result = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "run_mode": settings.run_mode,
              "embedding_backend": settings.embedding_backend, "model": settings.llm_model,
              "scope": "Public fixture smoke experiments, not broad benchmarks or production certification"}
    start = perf_counter()
    with tempfile.TemporaryDirectory(prefix="portfolio-eval-") as directory:
        isolated = settings.model_copy(update={"data_dir": Path(directory)})
        service = Services(isolated)
        principal = Principal("evaluation", "operator")
        try:
            session = service.store.new_session(principal, "Public evaluation")["id"]
            pdf = public_file("nist_ai_rmf")
            ingestion = service.ingest(principal, session, pdf.name, pdf.read_bytes())
            queries = [
                ("What are Govern, Map, Measure and Manage in AI risk management?", ["govern", "map", "measure", "manage"]),
                ("What characteristics make AI systems trustworthy?", ["valid", "reliable", "safe", "transparent"]),
                ("How should organizations manage AI risks?", ["risk", "organization"]),
            ]
            retrieval = []
            for query, expected in queries:
                begin = perf_counter()
                hits = service.rag.retrieve(principal, query, k=4)
                text = " ".join(hit["text"].lower() for hit in hits)
                retrieval.append({"query": query, "hits": len(hits), "expected_term_recall":
                    sum(term in text for term in expected)/len(expected), "elapsed_seconds": round(perf_counter()-begin, 3),
                    "pages": sorted({hit["page"] for hit in hits}), "valid_ids": all(hit["id"] for hit in hits)})
            result["retrieval"] = {"chunks": ingestion["chunks"], "pages": ingestion["pages"], "cases": retrieval,
                                    "metric_limit": "Lexical expected-term smoke check, not expert relevance or faithfulness evaluation"}
            result["ml"] = {}
            for name in ("uci_iris", "sklearn_diabetes"):
                begin = perf_counter()
                report = train_models(read_csv(public_file(name).read_bytes(), isolated), isolated)
                result["ml"][name] = {**report, "elapsed_seconds": round(perf_counter()-begin, 3)}
        finally:
            service.close()
    result["elapsed_seconds"] = round(perf_counter()-start, 3)
    path = ROOT / "reports" / "public-evaluation.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
