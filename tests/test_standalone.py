"""Assert the installed runtime and fixtures belong to this checkout."""
import json
from pathlib import Path

import portfolio
from portfolio.config import ROOT
from portfolio.public_data import public_file
from portfolio.ui import PROJECTS


def test_checkout_is_self_contained():
    manifest = json.loads((ROOT / "project.json").read_text(encoding="utf-8"))
    assert Path(portfolio.__file__).resolve().is_relative_to(ROOT)
    assert PROJECTS == [manifest["title"]]
    assert (ROOT / "app.py").is_file() and (ROOT / "demo.py").is_file()
    assert not (ROOT / "projects").exists()
    assert public_file("nist_ai_rmf").is_relative_to(ROOT)
    assert public_file("uci_iris").is_file()
