#!/usr/bin/env python3
# End-to-end check of cross-slot prefix sharing in llama-server (--kv-unified).
#
# Concurrent chat requests share a long system prompt. A leader request starts first and keeps generating, then the
# followers arrive: they should adopt the leader's cells for the system prompt instead of computing it (large cache_n,
# small prompt_n). Afterwards every follower is sent again on its own; greedy outputs should match the shared run.
#
# The system prompt starts with a random nonce, so no earlier cache holds it and the shared run cannot reuse RAM caches.
#
# Start the server with --kv-unified and --parallel above --n-followers, e.g. --parallel 8, then:
#   python3 scripts/check-server-prefix-share.py --url http://localhost:8100 --n-followers 4
# Run it again against a server started with LLAMA_PREFIX_SHARE=0 to compare prompt_n and time to first token.

import argparse
import difflib
import json
import random
import sys
import threading
import time
import urllib.request


def build_system_prompt(nonce: int, n_facts: int) -> str:
    rng = random.Random(nonce)
    colors = ["red", "green", "blue", "yellow", "purple", "orange", "black", "white"]
    things = ["bridge", "river", "tower", "garden", "library", "market", "harbor", "station"]
    lines = [f"Session {nonce}. You answer questions about the facts below. Reply with the fact only."]
    for i in range(n_facts):
        lines.append(f"Fact {i}: the {rng.choice(colors)} {rng.choice(things)} number {rng.randint(100, 999)} "
                     f"opens at {rng.randint(1, 12)} o'clock and holds {rng.randint(2, 90)} visitors.")
    return "\n".join(lines)


def chat(url: str, system: str, user: str, max_tokens: int, first_token: threading.Event | None = None) -> dict:
    body = {
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": first_token is not None,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    ttft = None
    with urllib.request.urlopen(req, timeout=3600) as resp:
        if first_token is None:
            data = json.loads(resp.read())
            return {"text": data["choices"][0]["message"].get("content") or "", "timings": data.get("timings", {}),
                    "seconds": time.time() - t0}

        text = []
        timings = {}
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[len("data: "):])
            delta = chunk["choices"][0]["delta"].get("content") if chunk.get("choices") else None
            if delta:
                if ttft is None:
                    ttft = time.time() - t0
                    first_token.set()
                text.append(delta)
            timings = chunk.get("timings", timings)
        first_token.set()
        return {"text": "".join(text), "timings": timings, "seconds": time.time() - t0, "ttft": ttft}


def metric(url: str, name: str) -> float:
    with urllib.request.urlopen(f"{url}/metrics", timeout=30) as resp:
        for line in resp.read().decode().splitlines():
            if line.startswith(f"llamacpp:{name} "):
                return float(line.split()[1])
    return float("nan")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8100")
    parser.add_argument("--n-followers", type=int, default=4)
    parser.add_argument("--n-facts", type=int, default=120)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--leader-tokens", type=int, default=400)
    args = parser.parse_args()

    nonce = random.randint(10**8, 10**9)
    system = build_system_prompt(nonce, args.n_facts)
    questions = [f"What is fact {(7 * i + 3) % args.n_facts}?" for i in range(args.n_followers + 1)]

    shares_before = metric(args.url, "prefix_shares_total")

    # shared run: the leader holds the system prompt in memory while the followers arrive
    first_token = threading.Event()
    results: list = [None] * (args.n_followers + 1)

    def run_leader() -> None:
        results[0] = chat(args.url, system, questions[0] + " Then list every fact again.", args.leader_tokens, first_token)

    def run_follower(k: int) -> None:
        results[k] = chat(args.url, system, questions[k], args.max_tokens)

    leader = threading.Thread(target=run_leader)
    leader.start()
    if not first_token.wait(timeout=600):
        print("leader produced no token")
        return 1

    followers = [threading.Thread(target=run_follower, args=(k,)) for k in range(1, args.n_followers + 1)]
    for t in followers:
        t.start()
    for t in followers:
        t.join()
    leader.join()

    shares_after = metric(args.url, "prefix_shares_total")

    # reference run: each follower alone
    references = [None] + [chat(args.url, system, questions[k], args.max_tokens) for k in range(1, args.n_followers + 1)]

    all_ok = True
    for k in range(1, args.n_followers + 1):
        r, ref = results[k], references[k]
        same = r["text"] == ref["text"]
        ratio = difflib.SequenceMatcher(None, r["text"], ref["text"], autojunk=False).ratio()
        all_ok = all_ok and same
        print(f"follower {k}: cache_n = {r['timings'].get('cache_n')}, prompt_n = {r['timings'].get('prompt_n')}, "
              f"prompt_ms = {r['timings'].get('prompt_ms', 0):.0f}, output {'identical' if same else f'differs ({ratio:.1%} similar)'}")
        if not same:
            print(f"    shared:    {r['text']!r}")
            print(f"    reference: {ref['text']!r}")

    print(f"leader: prompt_n = {results[0]['timings'].get('prompt_n')}, ttft = {results[0].get('ttft') or 0:.2f}s")
    print(f"prefix_shares_total: {shares_before:.0f} -> {shares_after:.0f}")
    print("ALL IDENTICAL" if all_ok else "SOME OUTPUTS DIFFER (batched decoding can drift, check the similarity)")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
