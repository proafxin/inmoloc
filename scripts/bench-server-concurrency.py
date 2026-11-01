#!/usr/bin/env python3
# Generation throughput and inter-token latency of local-inference-server at several concurrency levels.
#
# For each level, that many streaming chat requests with different prompts run at once. Reported per level:
#   agg tok/s  - generated tokens of all requests / wall time of the level (what the server delivers in total)
#   req tok/s  - mean generation speed of a single request, from the server timings
#   ttft       - mean time from when the server starts processing a request to its first token
#   wait       - mean time a request waited for that, from when it was sent
#   itl p50/p99/max - gaps between decoding steps of a request (tokens arriving within 2 ms count as one step)
#   accept     - draft tokens accepted / drafted, when speculative decoding is on
#
# Compare speculative decoding by running it against a server with and without --spec-type, e.g.
#   python3 scripts/bench-server-concurrency.py --url http://localhost:8100 --levels 1 2 4 8
#
# --prompt-tokens gives every request its own text of about that many tokens to continue, different in every request of
# every level (and for every --seed), so that no server can reuse a cached prompt; it takes a number or a range such as
# 64-2048: with a range, the lengths of a level are spread evenly over it, so that levels compare the same mix and every
# server gets the same requests. Every request generates up to the output limit of the server. --json writes the numbers
# of every level, for scripts/bench-suite.py

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


def run_request(url: str, prompt: str, out: dict) -> None:
    try:
        run_request_impl(url, prompt, out)
    except Exception as e:  # noqa: BLE001
        out["error"] = str(e)


def run_request_impl(url: str, prompt: str, out: dict) -> None:
    body = {
        "messages": [{"role": "user", "content": prompt}],
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
    first = arrivals[0] - t0 if arrivals else float("nan")
    out["ttft"] = timings.get("prompt_ms", float("nan"))/1000
    out["wait"] = first - out["ttft"]
    # tokens of one decoding step (e.g. accepted drafts) arrive as separate events within a moment: count one gap per step
    steps = [t for i, t in enumerate(arrivals) if i == 0 or t - arrivals[i - 1] > 0.002]
    out["gaps"] = [b - a for a, b in zip(steps, steps[1:])]
    out["timings"] = timings


def percentile(values: list, p: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    return values[min(len(values) - 1, int(p * len(values)))]


# a number of tokens, or a range such as 64-2048
def token_range(value: str) -> tuple:
    low, _, high = value.partition("-")
    return int(low), int(high or low)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8100")
    parser.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--prompt-tokens", type=token_range, default=(0, 0),
                        help="tokens of a unique text for every request, a number or a range; 0 = a short essay prompt")
    parser.add_argument("--json", default="", help="JSON file for the numbers of every level")
    parser.add_argument("--seed", type=int, default=0,
                        help="varies the requests; runs with the same seed send the same requests")
    args = parser.parse_args()

    results = []

    print(f"{'conc':>4} {'wall s':>7} {'agg tok/s':>9} {'req tok/s':>9} {'ttft s':>7} {'wait s':>7} {'itl p50':>8} "
          f"{'itl p99':>8} {'itl max':>8} {'accept':>7} {'failed':>6}")
    for level in args.levels:
        # the lengths are spread evenly over the range at every level, so that levels compare the same mix of requests;
        # shuffled, so that the short ones do not arrive first
        low, high = args.prompt_tokens
        lengths = [low + (high - low)*(2*r + 1)//(2*level) for r in range(level)]
        random.Random(f"size-{args.seed}-{level}").shuffle(lengths)

        prompts = []
        for r in range(level):
            prompt_tokens = lengths[r]
            if prompt_tokens > 0:
                rng = random.Random(f"{args.seed}-{level}-{r}")
                prompts.append(f"Continue this text about {rng.choice(TOPICS)} with a detailed essay.\n\n" +
                               long_document(prompt_tokens, rng.randrange(1 << 62)))
            else:
                prompts.append(f"Write a detailed essay about {TOPICS[r % len(TOPICS)]}.")

        outs = [dict() for _ in range(level)]

        threads = [threading.Thread(target=run_request, args=(args.url, prompts[r], outs[r]))
                   for r in range(level)]
        t0 = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.time() - t0

        good  = [o for o in outs if "error" not in o and o.get("timings")]
        n_bad = level - len(good)
        for r, o in enumerate(outs):
            if "error" in o:
                print(f"     request {r} failed: {o['error'][:120]}")

        def mean(values):
            values = [v for v in values if v == v]
            return statistics.mean(values) if values else float("nan")

        n_gen = sum(o["timings"].get("predicted_n", 0) for o in good)
        req_tps = mean([o["timings"].get("predicted_per_second", 0.0) for o in good])
        ttft = mean([o["ttft"] for o in good])
        wait = mean([o["wait"] for o in good])
        gaps = [g for o in good for g in o["gaps"]]
        n_draft = sum(o["timings"].get("draft_n", 0) for o in good)
        n_acc = sum(o["timings"].get("draft_n_accepted", 0) for o in good)
        accept = f"{n_acc / n_draft:.2f}" if n_draft else "-"
        print(f"{level:>4} {wall:>7.1f} {n_gen / wall:>9.1f} {req_tps:>9.1f} {ttft:>7.2f} {wait:>7.2f} "
              f"{percentile(gaps, 0.5) * 1e3:>6.0f}ms {percentile(gaps, 0.99) * 1e3:>6.0f}ms "
              f"{max(gaps, default=float('nan')) * 1e3:>6.0f}ms {accept:>7} {n_bad:>6}")

        results.append({
            "level": level, "wall_s": wall, "agg_tps": n_gen / wall, "req_tps": req_tps,
            "ttft_s": ttft, "wait_s": wait,
            "itl_p50_ms": percentile(gaps, 0.5) * 1e3, "itl_p99_ms": percentile(gaps, 0.99) * 1e3,
            "accept": n_acc / n_draft if n_draft else None, "failed": n_bad, "n_gen": n_gen,
            "n_prompt": sum(o["timings"].get("prompt_n", 0) for o in good),
        })

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)
    return 1 if any(r["failed"] for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
