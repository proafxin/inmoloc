import pytest
import requests
from utils import *

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server() -> None:
    global server
    server = ServerPreset.tinyllama2()


def test_server_start_simple() -> None:
    global server
    server.start()
    res = server.make_request("GET", "/health")
    assert res.status_code == 200


def test_server_props() -> None:
    global server
    server.start()
    res = server.make_request("GET", "/props")
    assert res.status_code == 200
    assert ".gguf" in res.body["model_path"]
    assert res.body["total_slots"] > 0  # as many as the elastic memory can admit, not a setting
    default_val = res.body["default_generation_settings"]
    assert default_val["n_ctx"] > 0  # the cells of the elastic memory
    assert default_val["params"]["seed"] == server.seed


def test_server_models() -> None:
    global server
    server.start()
    res = server.make_request("GET", "/models")
    assert res.status_code == 200
    assert len(res.body["data"]) == 1
    assert res.body["data"][0]["id"] == server.model_alias


def test_server_slots() -> None:
    global server

    # without slots endpoint enabled, this should return error
    server.server_slots = False
    server.start()
    res = server.make_request("GET", "/slots")
    assert res.status_code == 501  # ERROR_TYPE_NOT_SUPPORTED
    assert "error" in res.body
    server.stop()

    # with slots endpoint enabled, this should return slots info
    server.server_slots = True
    server.start()
    res = server.make_request("GET", "/slots")
    assert res.status_code == 200
    assert len(res.body) > 0
    assert res.body[0]["n_ctx"] > 0  # the cells of the elastic memory
    assert "params" not in res.body[0]


def test_load_split_model() -> None:
    global server
    server = ServerPreset.tinyllama_split()
    server.start()
    res = server.make_request(
        "POST",
        "/completion",
        data={
            "n_predict": 16,
            "prompt": "Hello",
            "temperature": 0.0,
        },
    )
    assert res.status_code == 200
    assert match_regex("(little|girl)+", res.body["content"])


def test_no_ui() -> None:
    global server
    # default: UI enabled
    server.start()
    url = f"http://{server.server_host}:{server.server_port}"
    res = requests.get(url)
    assert res.status_code == 200
    assert "<!doctype html>" in res.text
    server.stop()

    # with --no-ui, the UI should be disabled
    server.no_ui = True
    server.start()
    res = requests.get(url)
    assert res.status_code == 404


def test_server_model_aliases_and_tags() -> None:
    global server
    server.model_alias = "tinyllama-2,fim,code"
    server.model_tags = "chat,fim,small"
    server.start()
    res = server.make_request("GET", "/models")
    assert res.status_code == 200
    assert len(res.body["data"]) == 1
    model = res.body["data"][0]
    # aliases field must contain all aliases
    assert set(model["aliases"]) == {"tinyllama-2", "fim", "code"}
    # tags field must contain all tags
    assert set(model["tags"]) == {"chat", "fim", "small"}
    # id is derived from first alias (alphabetical order from std::set)
    assert model["id"] == "code"
