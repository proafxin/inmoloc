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
#
# --long N --long-tokens T gives N requests of each level a unique document of about T tokens to summarize, so long
# and short requests share the memory; ttft is then reported for both, and failed requests are counted

import argparse
import json
import random
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


WORDS = ("river stone market winter engine signal harbor letter garden copper season ledger bridge lantern "
         "orchard mirror valley thunder canvas anchor meadow compass silver timber beacon quarry").split()


# a document of about n_tokens tokens, different for every seed so that requests share no prefix
def long_document(n_tokens: int, seed: int) -> str:
    rng = random.Random(seed)
    words = [rng.choice(WORDS) for _ in range(int(n_tokens / 1.3))]
    lines = [" ".join(words[i:i + 12]) + f" (line {i // 12})." for i in range(0, len(words), 12)]
    return "\n".join(lines)


def run_request(url: str, prompt: str, max_tokens: int, out: dict) -> None:
    try:
        run_request_impl(url, prompt, max_tokens, out)
    except Exception as e:  # noqa: BLE001
        out["error"] = str(e)


def run_request_impl(url: str, prompt: str, max_tokens: int, out: dict) -> None:
    body = {
        "messages": [{"role": "user", "content": prompt}],
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
    pieces = []
    timings = {}
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[len("data: "):])
            if chunk.get("choices") and chunk["choices"][0]["delta"].get("content"):
                arrivals.append(time.time())
                pieces.append(chunk["choices"][0]["delta"]["content"])
            timings = chunk.get("timings", timings)
    out["ttft"] = arrivals[0] - t0 if arrivals else float("nan")
    # tokens of one decoding step (e.g. accepted drafts) arrive as separate events within a moment: count one gap per step
    steps = [t for i, t in enumerate(arrivals) if i == 0 or t - arrivals[i - 1] > 0.002]
    out["gaps"] = [b - a for a, b in zip(steps, steps[1:])]
    out["timings"] = timings
    out["text"] = "".join(pieces)


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
    parser.add_argument("--long", type=int, default=0, help="requests per level with a long document")
    parser.add_argument("--long-tokens", type=int, default=16000, help="approximate tokens of a long document")
    parser.add_argument("--save", default="", help="JSON file for the text of every response, to compare servers")
    parser.add_argument("--seed", type=int, default=0,
                        help="varies the long documents; runs with the same seed send the same documents, which the "
                             "server may then reuse from its cache")
    args = parser.parse_args()

    saved = {}

    print(f"{'conc':>4} {'wall s':>7} {'agg tok/s':>9} {'req tok/s':>9} {'ttft s':>7} {'ttft long':>9} {'itl p50':>8} "
          f"{'itl p99':>8} {'itl max':>8} {'accept':>7} {'failed':>6}")
    for level in args.levels:
        n_long = min(args.long, level)
        prompts = []
        for i in range(level):
            if i < n_long:
                prompts.append("Summarize the following document in detail.\n\n" + long_document(args.long_tokens, 1000000*args.seed + 1000*level + i))
            else:
                prompts.append(f"Write a detailed essay about {TOPICS[i % len(TOPICS)]}.")

        outs = [dict() for _ in range(level)]
        threads = [threading.Thread(target=run_request, args=(args.url, prompts[i], args.max_tokens, outs[i]))
                   for i in range(level)]
        t0 = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.time() - t0

        good  = [o for o in outs if "error" not in o and o.get("timings")]
        n_bad = level - len(good)
        for i, o in enumerate(outs):
            if "error" in o:
                print(f"     request {i} ({'long' if i < n_long else 'short'}) failed: {o['error'][:120]}")

        def mean(values):
            values = [v for v in values if v == v]
            return statistics.mean(values) if values else float("nan")

        n_gen = sum(o["timings"].get("predicted_n", 0) for o in good)
        req_tps = mean([o["timings"].get("predicted_per_second", 0.0) for o in good])
        ttft = mean([outs[i]["ttft"] for i in range(n_long, level) if outs[i] in good])
        ttft_long = mean([outs[i]["ttft"] for i in range(n_long) if outs[i] in good])
        gaps = [g for o in good for g in o["gaps"]]
        n_draft = sum(o["timings"].get("draft_n", 0) for o in good)
        n_acc = sum(o["timings"].get("draft_n_accepted", 0) for o in good)
        accept = f"{n_acc / n_draft:.2f}" if n_draft else "-"
        print(f"{level:>4} {wall:>7.1f} {n_gen / wall:>9.1f} {req_tps:>9.1f} {ttft:>7.2f} {ttft_long:>9.2f} "
              f"{percentile(gaps, 0.5) * 1e3:>6.0f}ms {percentile(gaps, 0.99) * 1e3:>6.0f}ms "
              f"{max(gaps, default=float('nan')) * 1e3:>6.0f}ms {accept:>7} {n_bad:>6}")

        saved[str(level)] = [o.get("text") for o in outs]

    if args.save:
        with open(args.save, "w") as f:
            json.dump(saved, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
