"""Attributed public fixtures; MCP accepts catalog IDs rather than arbitrary URLs."""
from __future__ import annotations

from io import BytesIO
import hashlib
import json
from pathlib import Path
import zipfile

import httpx
import pandas as pd
from sklearn.datasets import load_diabetes

from .config import ROOT

PUBLIC = ROOT / "assets" / "public"
CATALOG = {
    "nist_ai_rmf": {"name": "NIST AI Risk Management Framework 1.0", "kind": "pdf",
                    "url": "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf", "file": "nist-ai-rmf-1.0.pdf",
                    "citation": "NIST AI 100-1 (January 2023), DOI 10.6028/NIST.AI.100-1",
                    "rights": "U.S. government publication; retain original notices/attribution; no NIST endorsement"},
    "uci_iris": {"name": "UCI Iris", "kind": "csv",
                 "url": "https://archive.ics.uci.edu/static/public/53/iris.zip", "file": "iris.csv",
                 "citation": "Fisher, R. A. (1936). Iris. UCI Machine Learning Repository. DOI 10.24432/C56C76",
                 "rights": "CC BY 4.0, https://creativecommons.org/licenses/by/4.0/"},
    "sklearn_diabetes": {"name": "scikit-learn diabetes regression data", "kind": "csv",
                         "url": "https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset",
                         "file": "diabetes.csv", "citation": "Efron et al. (2004), Least Angle Regression; scikit-learn load_diabetes",
                         "rights": "Bundled public educational dataset; preserve dataset documentation and attribution"},
}


def download(url: str, limit=8*1024*1024) -> bytes:
    if url not in {item["url"] for item in CATALOG.values()}:
        raise ValueError("Only allowlisted public catalog URLs may be downloaded")
    chunks, size = [], 0
    with httpx.Client(timeout=25, follow_redirects=False) as client:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            if response.status_code != 200:
                raise ValueError("Unexpected download status/redirect")
            for chunk in response.iter_bytes(65536):
                size += len(chunk)
                if size > limit:
                    raise ValueError("Public download exceeds the configured size budget")
                chunks.append(chunk)
    return b"".join(chunks)


def bootstrap() -> dict:
    PUBLIC.mkdir(parents=True, exist_ok=True)
    records = []
    for key, item in CATALOG.items():
        target = PUBLIC / item["file"]
        if not target.exists():
            if key == "sklearn_diabetes":
                frame = load_diabetes(as_frame=True).frame.rename(columns={"target": "disease_progression"})
                content = frame.to_csv(index=False).encode()
            elif key == "uci_iris":
                archive = zipfile.ZipFile(BytesIO(download(item["url"])))
                member = next(info for info in archive.infolist() if info.filename == "iris.data")
                if member.file_size > 1000000:
                    raise ValueError("Unexpected Iris dataset member size")
                frame = pd.read_csv(BytesIO(archive.read(member)), header=None,
                                    names=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"])
                content = frame.dropna(how="all").to_csv(index=False).encode()
            else:
                content = download(item["url"])
                if not content.startswith(b"%PDF-"):
                    raise ValueError("Catalog PDF download did not return a PDF")
            temporary = target.with_suffix(target.suffix + ".partial")
            temporary.write_bytes(content)
            temporary.replace(target)
        records.append({"id": key, **item, "bytes": target.stat().st_size,
                        "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
    manifest = {"sources": records, "method": "Allowlisted HTTPS downloads; diabetes exported from sklearn's bundled public data"}
    (PUBLIC / "PROVENANCE.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def public_file(source_id: str) -> Path:
    if source_id not in CATALOG:
        raise ValueError("Unknown catalog ID")
    path = PUBLIC / CATALOG[source_id]["file"]
    if not path.is_file():
        raise ValueError("Public fixture missing; run uv run portfolio bootstrap")
    return path
