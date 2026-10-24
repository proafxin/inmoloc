import math

import pytest
from utils import *

server = ServerPreset.tinyllama2()

N_REQUESTS = 4
N_PREDICT = 200

# the most a probability of the two top tokens may differ by where a request in a batch picks another token than the
# same request served alone: the kernels pick by batch size and round differently, which only matters at a near-tie
NEAR_TIE = 0.01


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    # 4 requests of about 210 tokens do not fit 512 cells at once: some are preempted and resumed
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


# the generated tokens, and for each one the two most likely tokens with their probabilities
class Generation:
    def __init__(self):
        self.content = ""
        self.ids: list[int] = []
        self.top2: list[list[tuple[int, float]]] = []
        self.n_predicted = 0

    def add(self, probs: list[dict]):
        for p in probs:
            self.ids.append(p["id"])
            top = sorted(((t["id"], math.exp(t["logprob"])) for t in p["top_logprobs"]), key=lambda x: -x[1])
            self.top2.append(top[:2])


def complete(prompt: str, stream: bool, cache_prompt: bool = False) -> Generation:
    data = {"prompt": prompt, "n_predict": N_PREDICT, "temperature": 0.0, "ignore_eos": True, "cache_prompt": cache_prompt,
            "n_probs": 2}
    res = Generation()
    if not stream:
        r = server.make_request("POST", "/completion", data=data)
        assert r.status_code == 200
        res.content     = r.body["content"]
        res.n_predicted = r.body["tokens_predicted"]
        res.add(r.body["completion_probabilities"])
        return res

    for chunk in server.make_stream_request("POST", "/completion", data={**data, "stream": True}):
        res.content += chunk.get("content", "")
        res.add(chunk.get("completion_probabilities", []))
        if chunk.get("stop"):
            res.n_predicted = chunk["tokens_predicted"]
    return res


# the request generates the same tokens as when it is served alone, up to a near-tie: a token where the two most likely
# tokens are about as likely in both runs, after which the texts may differ; a preemption that drops, repeats or
# corrupts tokens makes the request differ where the model is sure of its token
def assert_same_up_to_near_tie(k: int, got: Generation, ref: Generation):
    assert got.n_predicted == N_PREDICT and len(got.ids) == N_PREDICT, f"request {k} generated {len(got.ids)} tokens"

    n = next((i for i in range(N_PREDICT) if got.ids[i] != ref.ids[i]), None)
    if n is None:
        assert got.content == ref.content, f"request {k} has the same tokens as the same request served alone but another text"
        return

    for name, gen in (("alone", ref), ("in the batch", got)):
        top = gen.top2[n]
        assert len(top) == 2 and top[0][1] - top[1][1] <= NEAR_TIE, \
            f"request {k} differs from the same request served alone at token {n}, where the request {name} is sure of its token: {top}"
    assert got.ids[n] in (t[0] for t in ref.top2[n]), \
        f"request {k} picks token {got.ids[n]} at token {n}, which is not one of the two most likely of the request served alone"


# a preempted request continues exactly where it stopped: in full, and the same as when it runs alone up to a near-tie,
# and a stream neither repeats nor drops text across the preemption
def test_preempt_and_resume():
    global server
    server.start()

    preemptions = metric("preemptions_total")
    resumes = metric("resumes_total")

    results = parallel_function_calls([(complete, (PROMPTS[k], k % 2 == 1)) for k in range(N_REQUESTS)])

    assert metric("preemptions_total") > preemptions
    assert metric("resumes_total") > resumes

    for k, got in enumerate(results):
        assert_same_up_to_near_tie(k, got, complete(PROMPTS[k], stream=False))


# with a prompt cache in host memory, a preempted request saves its state there and the resume restores it
def test_preempt_and_restore():
    global server
    server.cache_ram = 64
    server.start()

    restored = metric("restored_tokens_total")

    results = parallel_function_calls([(complete, (PROMPTS[k], k % 2 == 1, True)) for k in range(N_REQUESTS)])

    assert metric("restored_tokens_total") > restored

    for k, got in enumerate(results):
        assert_same_up_to_near_tie(k, got, complete(PROMPTS[k], stream=False))
