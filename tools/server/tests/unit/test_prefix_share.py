import pytest
from utils import *

server = ServerPreset.tinyllama2()

# the shared prefix must gain at least --prefix-share-min tokens (default 256)
N_PREFIX = 320


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.n_ctx = 1024
    server.n_slots = 2
    server.n_predict = 8
    server.kv_unified = True
    server.server_metrics = True
    server.temperature = 0.0
    server.cache_ram = 0  # no RAM prompt cache: the cells come from the slots alone


def metric(name: str) -> float:
    res = server.make_request("GET", "/metrics")
    assert res.status_code == 200
    for line in res.body.splitlines():
        if line.startswith(f"local_inference:{name} "):
            return float(line.split(" ", 1)[1])
    raise AssertionError(f"metric {name} not found")


def build_prefix() -> str:
    sentences = [
        "Once upon a time, there was a little girl named Lily.",
        "She liked to play in the park with her dog.",
        "One day, the sun was hot and the sky was blue.",
        "Lily saw a big tree and a small bird on it.",
    ]
    text = ""
    k = 0
    while True:
        text += " " + sentences[k % len(sentences)]
        k += 1
        res = server.make_request("POST", "/tokenize", data={"content": text})
        if len(res.body["tokens"]) >= N_PREFIX:
            return text


def complete(prompt: str, id_slot: int, cache_prompt: bool = True) -> dict:
    res = server.make_request("POST", "/completion", data={
        "prompt": prompt,
        "id_slot": id_slot,
        "cache_prompt": cache_prompt,
        "temperature": 0.0,
    })
    assert res.status_code == 200
    return res.body


# a request on another slot starts from the cells of the slot that holds the same prefix, and its output is the same as
# when the prompt is computed in full
def test_share_prefix_from_idle_slot():
    global server
    server.start()
    prefix = build_prefix()

    complete(prefix + " The end.", id_slot=0)

    shares = metric("prefix_shares_total")
    res = complete(prefix + " Then her dog ran to the tree.", id_slot=1)

    assert metric("prefix_shares_total") == shares + 1
    assert res["timings"]["cache_n"] >= N_PREFIX - 8
    assert res["timings"]["prompt_n"] < 32

    reference = complete(prefix + " Then her dog ran to the tree.", id_slot=0, cache_prompt=False)
    assert res["content"] == reference["content"]


def test_no_prefix_share():
    global server
    server.no_prefix_share = True
    server.start()
    prefix = build_prefix()

    complete(prefix + " The end.", id_slot=0)

    shares = metric("prefix_shares_total")
    res = complete(prefix + " Then her dog ran to the tree.", id_slot=1)

    assert metric("prefix_shares_total") == shares
    assert res["timings"]["prompt_n"] >= N_PREFIX
