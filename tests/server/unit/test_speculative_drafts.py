import os

import pytest
from utils import (
    DFLASH_DRAFT_FILE_URL,
    EAGLE3_DRAFT_CONVERT,
    EAGLE3_DRAFT_FILE,
    ServerPreset,
    ServerProcess,
    download_file,
    parallel_function_calls,
)

# The drafts that read the target's hidden states (DFlash, EAGLE3): their layers are kept in the elastic memory of the
# target, next to the target's own. Greedy decoding gives the target's tokens with and without the draft.

DRAFTS = [
    # (preset of the target, draft file: a URL of TEST_FILE_URLS or a file of the cache, spec type)
    ("qwen3_4b", DFLASH_DRAFT_FILE_URL, "draft-dflash"),
    ("qwen3_4b", EAGLE3_DRAFT_FILE,     "draft-eagle3"),
]


def draft_path(draft: str) -> str:
    if draft.startswith("https://"):
        return download_file(draft)
    if not os.path.exists(draft):
        pytest.fail(f"{draft} is missing: convert the draft once, from tests/server:\n{EAGLE3_DRAFT_CONVERT}")
    return draft

REQUEST = {
    "prompt": "The capital of France is Paris. The capital of Germany is",
    "temperature": 0.0,
    "seed": 42,
    "n_predict": 32,
    "return_tokens": True,
}


def make_server(preset: str, draft_url: str | None, spec_type: str | None) -> ServerProcess:
    server = getattr(ServerPreset, preset)()
    if draft_url is not None:
        server.model_draft = draft_path(draft_url)
        server.spec_type = spec_type
        server.spec_draft_n_max = 8
    return server


@pytest.mark.parametrize(("preset", "draft_url", "spec_type"), DRAFTS)
def test_draft_matches_target(preset: str, draft_url: str, spec_type: str) -> None:
    server = make_server(preset, None, None)
    server.start()
    res = server.make_request("POST", "/completion", data=REQUEST)
    assert res.status_code == 200
    tokens_target = res.body["tokens"]
    server.stop()

    server = make_server(preset, draft_url, spec_type)
    server.start()
    res = server.make_request("POST", "/completion", data=REQUEST)
    assert res.status_code == 200
    assert res.body["timings"]["draft_n"] > 0
    assert res.body["timings"]["draft_n_accepted"] > 0
    assert res.body["tokens"] == tokens_target
    server.stop()


@pytest.mark.parametrize(("preset", "draft_url", "spec_type"), DRAFTS)
def test_draft_parallel_requests(preset: str, draft_url: str, spec_type: str) -> None:
    # several requests share the elastic memory, the draft layers of each in its own blocks
    server = make_server(preset, draft_url, spec_type)
    server.n_slots = 2
    server.start()

    tasks = [
        (server.make_request, ("POST", "/completion", {**REQUEST, "prompt": prompt}))
        for prompt in ["The capital of Italy is", "Once upon a time, there was"]
    ]
    results = parallel_function_calls(tasks)
    for res in results:
        assert res.status_code == 200
        assert res.body["timings"]["draft_n"] > 0
    server.stop()
