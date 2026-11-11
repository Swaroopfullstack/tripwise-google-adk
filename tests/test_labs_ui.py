import math
import os
import sqlite3
import subprocess
import sys
from time import monotonic, sleep

import httpx
import pytest

from portfolio.config import ROOT
from portfolio.labs import cosine_lab, owned_sql_lab, prompts_and_history_lab, sampling_lab, sequential_interrupt_lab, tool_graph_lab


def test_numeric_sampling_and_cosine():
    result = sampling_lab()
    assert math.isclose(sum(result["probabilities"].values()), 1)
    assert math.isclose(sum(result["normalized_candidates"].values()), 1)
    assert len(result["normalized_candidates"]) <= 3
    assert math.isclose(cosine_lab()["cosine_similarity"], .8)


@pytest.mark.parametrize("query", ["DELETE FROM owned_orders", "SELECT * FROM sqlite_master", "ATTACH DATABASE 'private.sqlite' AS other", "PRAGMA database_list"])
def test_read_only_sql_denies_mutation_and_other_tables(query):
    with pytest.raises(sqlite3.DatabaseError):
        owned_sql_lab(query)


def test_actual_lcel_tool_graph_streams_and_interrupt_order():
    assert len(owned_sql_lab()["rows"]) == 2
    result = tool_graph_lab()
    assert result["answer"] == "19" and result["observation_ids"] == ["addition-observation"]
    assert "custom" in result["stream_modes"] and result["checkpoints"] > 2 and not result["next"]
    form = sequential_interrupt_lab()
    assert form["first"]["field"] == "name" and form["second"]["field"] == "purpose"
    assert form["final"]["name"] == "Learner"
    assert prompts_and_history_lab()["trimmed_count"] == 2


def test_gradio_interface_and_fixture_stream():
    pytest.importorskip("gradio")
    from portfolio.gradio_app import create_demo
    from portfolio.foundations import stream_reply
    interface = create_demo()
    assert interface is not None
    parts = list(stream_reply("Explain checkpoints", []))
    assert len(parts) > 1 and "fixture" in parts[-1].lower()


def test_streamlit_against_actual_http_api(tmp_path):
    """Real Uvicorn socket + Streamlit's execution harness, no mocked backend."""
    from streamlit.testing.v1 import AppTest
    import socket
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    origin = f"http://127.0.0.1:{port}"
    env = {**os.environ, "DATA_DIR": str(tmp_path / "ui-data"), "APP_ENV": "test", "RUN_MODE": "demo",
           "EMBEDDING_BACKEND": "hashing", "REQUESTS_PER_MINUTE": "300", "PYTHONUTF8": "1"}
    process = subprocess.Popen([sys.executable, "-B", "-X", "utf8", "-m", "uvicorn", "portfolio.api:app", "--host", "127.0.0.1", "--port", str(port)],
                               cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    previous = os.environ.get("API_BASE_URL")
    os.environ["API_BASE_URL"] = origin
    try:
        deadline = monotonic()+90
        while monotonic() < deadline:
            try:
                if httpx.get(origin+"/health/live", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            if process.poll() is not None:
                raise AssertionError("API exited during UI integration startup")
            sleep(.25)
        else:
            raise AssertionError("API startup deadline")
        app = AppTest.from_file(str(ROOT/"app.py"), default_timeout=60).run()
        assert not app.exception
        assert app.title[0].value == 'TripWise ADK'
        assert not app.error
        assert all(item.label != "Project" for item in app.sidebar.selectbox)
    finally:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        if previous is None:
            os.environ.pop("API_BASE_URL", None)
        else:
            os.environ["API_BASE_URL"] = previous
