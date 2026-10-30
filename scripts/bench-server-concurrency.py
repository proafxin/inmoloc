#!/usr/bin/env python3
# Generation throughput and inter-token latency of local-inference-server at several concurrency levels.
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
#
# --prompt-tokens P gives every short request its own text of about P tokens to continue, different in every request of
# every level (and for every --seed), so that no server can reuse a cached prompt; --per-client K makes each of the
# clients of a level send K requests one after another, so that a level is a sustained load and not one burst
#
# --prompt-tokens-max and --max-tokens-max make the requests differ in size: each request draws its prompt length from
# [--prompt-tokens, --prompt-tokens-max] and its output length from [--max-tokens, --max-tokens-max], fixed by --seed so
# that every server gets the same requests; --json writes the numbers of every level, for scripts/bench-suite.py

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
    parser.add_argument("--max-tokens-max", type=int, default=0,
                        help="if larger than --max-tokens, each request draws its output length from the range")
    parser.add_argument("--long", type=int, default=0, help="requests per level with a long document")
    parser.add_argument("--long-tokens", type=int, default=16000, help="approximate tokens of a long document")
    parser.add_argument("--prompt-tokens", type=int, default=0,
                        help="approximate tokens of a unique text for every short request, 0 = a short essay prompt")
    parser.add_argument("--prompt-tokens-max", type=int, default=0,
                        help="if larger than --prompt-tokens, each request draws its prompt length from the range")
    parser.add_argument("--json", default="", help="JSON file for the numbers of every level")
    parser.add_argument("--per-client", type=int, default=1, help="requests each client sends one after another")
    parser.add_argument("--save", default="", help="JSON file for the text of every response, to compare servers")
    parser.add_argument("--seed", type=int, default=0,
                        help="varies the long documents; runs with the same seed send the same documents, which the "
                             "server may then reuse from its cache")
    args = parser.parse_args()

    saved = {}
    results = []

    print(f"{'conc':>4} {'wall s':>7} {'agg tok/s':>9} {'req tok/s':>9} {'ttft s':>7} {'ttft long':>9} {'itl p50':>8} "
          f"{'itl p99':>8} {'itl max':>8} {'accept':>7} {'failed':>6}")
    for level in args.levels:
        n_long = min(args.long, level)
        n_req  = level*args.per_client

        # request r is the (r // level)-th of client r % level; the clients below n_long send long documents
        def is_long(r):
            return r % level < n_long

        prompts = []
        max_tokens = []
        for r in range(n_req):
            rng_size = random.Random(f"size-{args.seed}-{level}-{r}")
            max_tokens.append(rng_size.randint(args.max_tokens, args.max_tokens_max)
                              if args.max_tokens_max > args.max_tokens else args.max_tokens)
            prompt_tokens = (rng_size.randint(args.prompt_tokens, args.prompt_tokens_max)
                             if args.prompt_tokens_max > args.prompt_tokens else args.prompt_tokens)
            i = r % level
            if is_long(r):
                prompts.append("Summarize the following document in detail.\n\n" +
                               long_document(args.long_tokens, 1000000*args.seed + 1000*level + i + 1000000000*(r // level)))
            elif prompt_tokens > 0:
                rng = random.Random(f"{args.seed}-{level}-{r}")
                prompts.append(f"Continue this text about {rng.choice(TOPICS)} with a detailed essay.\n\n" +
                               long_document(prompt_tokens, rng.randrange(1 << 62)))
            else:
                prompts.append(f"Write a detailed essay about {TOPICS[i % len(TOPICS)]}.")

        outs = [dict() for _ in range(n_req)]

        def client(i):
            for r in range(i, n_req, level):
                run_request(args.url, prompts[r], max_tokens[r], outs[r])

        threads = [threading.Thread(target=client, args=(i,)) for i in range(level)]
        t0 = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.time() - t0

        good  = [o for o in outs if "error" not in o and o.get("timings")]
        n_bad = n_req - len(good)
        for r, o in enumerate(outs):
            if "error" in o:
                print(f"     request {r} ({'long' if is_long(r) else 'short'}) failed: {o['error'][:120]}")

        def mean(values):
            values = [v for v in values if v == v]
            return statistics.mean(values) if values else float("nan")

        n_gen = sum(o["timings"].get("predicted_n", 0) for o in good)
        req_tps = mean([o["timings"].get("predicted_per_second", 0.0) for o in good])
        ttft = mean([outs[r]["ttft"] for r in range(n_req) if not is_long(r) and outs[r] in good])
        ttft_long = mean([outs[r]["ttft"] for r in range(n_req) if is_long(r) and outs[r] in good])
        gaps = [g for o in good for g in o["gaps"]]
        n_draft = sum(o["timings"].get("draft_n", 0) for o in good)
        n_acc = sum(o["timings"].get("draft_n_accepted", 0) for o in good)
        accept = f"{n_acc / n_draft:.2f}" if n_draft else "-"
        print(f"{level:>4} {wall:>7.1f} {n_gen / wall:>9.1f} {req_tps:>9.1f} {ttft:>7.2f} {ttft_long:>9.2f} "
              f"{percentile(gaps, 0.5) * 1e3:>6.0f}ms {percentile(gaps, 0.99) * 1e3:>6.0f}ms "
              f"{max(gaps, default=float('nan')) * 1e3:>6.0f}ms {accept:>7} {n_bad:>6}")

        saved[str(level)] = [o.get("text") for o in outs]

        results.append({
            "level": level, "wall_s": wall, "agg_tps": n_gen / wall, "req_tps": req_tps,
            "ttft_s": ttft, "ttft_long_s": ttft_long,
            "itl_p50_ms": percentile(gaps, 0.5) * 1e3, "itl_p99_ms": percentile(gaps, 0.99) * 1e3,
            "accept": n_acc / n_draft if n_draft else None, "failed": n_bad, "n_gen": n_gen,
            "n_prompt": sum(o["timings"].get("prompt_n", 0) for o in good),
        })

    if args.save:
        with open(args.save, "w") as f:
            json.dump(saved, f)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)
    return 1 if any(r["failed"] for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
