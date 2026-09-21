#!/usr/bin/env python3
# Generation throughput and inter-token latency of llama-server at several concurrency levels.
#
# For each level, that many streaming chat requests with different prompts run at once. Reported per level:
#   agg tok/s  - generated tokens of all requests / wall time of the level (what the server delivers in total)
#   req tok/s  - mean generation speed of a single request, from the server timings
#   ttft       - mean time to first token
#   itl p50/p99/max - gaps between decoding steps of a request (tokens arriving within 2 ms count as one step)
#   accept     - draft tokens accepted / drafted, when speculative decoding is on
#
# Compare speculative decoding by running it against a server with and without --spec-type, e.g.
#   python3 scripts/bench-server-concurrency.py --url http://localhost:8100 --levels 1 2 4 8 --max-tokens 512

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.request

TOPICS = [
    "the history of lighthouses", "how bread rises", "the life cycle of stars", "building a wooden boat",
    "the migration of arctic terns", "how compilers optimize loops", "the rules of chess openings",
    "the chemistry of coffee roasting", "the design of suspension bridges", "how vaccines train immunity",
    "the economics of container shipping", "the geology of volcanic islands", "writing a symphony",
    "the invention of the printing press", "caring for bonsai trees", "how noise cancelling headphones work",
]


def run_request(url: str, topic: str, max_tokens: int, out: dict) -> None:
    body = {
        "messages": [{"role": "user", "content": f"Write a detailed essay about {topic}."}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
        "ignore_eos": True,
    }
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    arrivals = []
    timings = {}
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[len("data: "):])
            if chunk.get("choices") and chunk["choices"][0]["delta"].get("content"):
                arrivals.append(time.time())
            timings = chunk.get("timings", timings)
    out["ttft"] = arrivals[0] - t0 if arrivals else float("nan")
    # tokens of one decoding step (e.g. accepted drafts) arrive as separate events within a moment: count one gap per step
    steps = [t for i, t in enumerate(arrivals) if i == 0 or t - arrivals[i - 1] > 0.002]
    out["gaps"] = [b - a for a, b in zip(steps, steps[1:])]
    out["timings"] = timings


def percentile(values: list, p: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    return values[min(len(values) - 1, int(p * len(values)))]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8100")
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    print(f"{'conc':>4} {'agg tok/s':>9} {'req tok/s':>9} {'ttft s':>7} {'itl p50':>8} {'itl p99':>8} {'itl max':>8} {'accept':>7}")
    for level in args.levels:
        outs = [dict() for _ in range(level)]
        threads = [threading.Thread(target=run_request, args=(args.url, TOPICS[i % len(TOPICS)], args.max_tokens, outs[i]))
                   for i in range(level)]
        t0 = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.time() - t0

        n_gen = sum(o["timings"].get("predicted_n", 0) for o in outs)
        req_tps = statistics.mean(o["timings"].get("predicted_per_second", 0.0) for o in outs)
        ttft = statistics.mean(o["ttft"] for o in outs)
        gaps = [g for o in outs for g in o["gaps"]]
        n_draft = sum(o["timings"].get("draft_n", 0) for o in outs)
        n_acc = sum(o["timings"].get("draft_n_accepted", 0) for o in outs)
        accept = f"{n_acc / n_draft:.2f}" if n_draft else "-"
        print(f"{level:>4} {n_gen / wall:>9.1f} {req_tps:>9.1f} {ttft:>7.2f} {percentile(gaps, 0.5) * 1e3:>6.0f}ms "
              f"{percentile(gaps, 0.99) * 1e3:>6.0f}ms {max(gaps, default=float('nan')) * 1e3:>6.0f}ms {accept:>7}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
