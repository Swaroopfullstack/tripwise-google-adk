import json

from fastapi.testclient import TestClient

from portfolio.api import create_app


def test_actual_graph_progress_over_sse_with_owned_result(settings):
    with TestClient(create_app(settings)) as client:
        headers = {"Authorization": "Bearer demo-alice-key"}
        session = client.post("/v1/sessions", json={"title": "SSE"}, headers=headers).json()["id"]
        body = {"session_id": session, "message": "calculate 6 * 7"}
        assert client.post("/v1/chat/stream", json=body).status_code == 401
        with client.stream("POST", "/v1/chat/stream", json=body, headers=headers) as response:
            assert response.status_code == 200
            lines = list(response.iter_lines())
        assert "event: progress" in lines and "event: result" in lines and "event: error" not in lines
        result_line = lines[lines.index("event: result")+1]
        result = json.loads(result_line.removeprefix("data: "))
        assert "42" in result["answer"] and result["route"]["route"] == "tools"
        assert "messages" not in result and "generation" not in result
        denied = client.post("/v1/chat/stream", json=body, headers={"Authorization": "Bearer demo-bob-key"})
        assert denied.status_code == 404
