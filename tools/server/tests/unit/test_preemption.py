import pytest
from utils import *

server = ServerPreset.tinyllama2()

N_REQUESTS = 4
N_PREDICT = 200


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    # 4 requests of about 210 tokens do not fit 512 cells at once: the newest are preempted and resumed
    server.n_ctx = 512
    server.n_slots = N_REQUESTS
    server.n_predict = N_PREDICT
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


PROMPTS = [
    "Once upon a time, there was a little girl named Lily.",
    "One day, a big dog went to the park with his friend.",
    "Tom had a red ball. He liked to play with it every day.",
    "The sun was shining and the birds were singing in the tree.",
]


def complete(prompt: str, stream: bool) -> tuple[str, int]:
    data = {"prompt": prompt, "n_predict": N_PREDICT, "temperature": 0.0, "ignore_eos": True, "cache_prompt": False}
    if not stream:
        res = server.make_request("POST", "/completion", data=data)
        assert res.status_code == 200
        return res.body["content"], res.body["tokens_predicted"]

    content = ""
    n_predicted = 0
    for chunk in server.make_stream_request("POST", "/completion", data={**data, "stream": True}):
        content += chunk.get("content", "")
        if chunk.get("stop"):
            n_predicted = chunk["tokens_predicted"]
    return content, n_predicted


# a preempted request continues exactly where it stopped: the same output as when it runs alone, in full, and a stream
# neither repeats nor drops text across the preemption
def test_preempt_and_resume():
    global server
    server.start()

    preemptions = metric("preemptions_total")
    resumes = metric("resumes_total")

    results = parallel_function_calls([(complete, (PROMPTS[k], k % 2 == 1)) for k in range(N_REQUESTS)])

    assert metric("preemptions_total") > preemptions
    assert metric("resumes_total") > resumes

    for k, (content, n_predicted) in enumerate(results):
        assert n_predicted == N_PREDICT
        reference, _ = complete(PROMPTS[k], stream=False)
        assert content == reference, f"request {k} differs from the same request served alone"
