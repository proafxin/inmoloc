#!/usr/bin/env python3
# End-to-end check of KV-cache preemption in llama-server (--kv-unified).
#
# Sends more concurrent requests than the unified KV cache can hold, so the server has to preempt and resume.
# Each request continues a count ("10001\n10002\n..."); a correct resume continues exactly where the generation
# stopped, so every generated line must be the next number. Half of the requests stream, to check that a resumed
# stream neither repeats nor drops text.
#
# Start the server with a small context, e.g. --ctx-size 8192 --parallel 6 --kv-unified, then:
#   python3 scripts/check-server-preemption.py --url http://localhost:8100 --n-requests 6 --n-predict 3000

import argparse
import json
import sys
import threading
import time
import urllib.request


def build_prompt(start: int, n_lines: int) -> str:
    return "".join(f"{start + i}\n" for i in range(n_lines))


def post(url: str, body: dict, stream: bool) -> tuple[str, dict]:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as resp:
        if not stream:
            data = json.loads(resp.read())
            return data["content"], data

        text = []
        last = {}
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            payload = line[len("data: "):]
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            if "error" in chunk:
                raise RuntimeError(chunk["error"])
            text.append(chunk.get("content", ""))
            last = chunk
        return "".join(text), last


def check_count(text: str, first_expected: int) -> tuple[int, str]:
    # returns the number of correct consecutive lines and a description of the first mismatch ("" if none)
    lines = text.split("\n")
    # the last line can be cut by n_predict
    complete, tail = lines[:-1], lines[-1]
    expected = first_expected
    for i, line in enumerate(complete):
        if line != str(expected):
            return i, f"line {i}: expected {expected!r}, got {line!r}"
        expected += 1
    if tail and not str(expected).startswith(tail):
        return len(complete), f"last line: expected a prefix of {expected!r}, got {tail!r}"
    return len(complete), ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8100")
    parser.add_argument("--n-requests", type=int, default=6)
    parser.add_argument("--n-predict", type=int, default=3000)
    parser.add_argument("--n-prompt-lines", type=int, default=20)
    args = parser.parse_args()

    results = [None] * args.n_requests

    def run(k: int) -> None:
        start = (k + 1) * 10000 + 1
        stream = k % 2 == 1
        body = {
            "prompt": build_prompt(start, args.n_prompt_lines),
            "n_predict": args.n_predict,
            "temperature": 0.0,
            "ignore_eos": True,
            "cache_prompt": False,
            "stream": stream,
        }
        t0 = time.time()
        try:
            text, info = post(f"{args.url}/completion", body, stream)
            n_ok, mismatch = check_count(text, start + args.n_prompt_lines)
            results[k] = {
                "stream": stream,
                "ok": mismatch == "",
                "n_lines_ok": n_ok,
                "mismatch": mismatch,
                "n_predicted": info.get("tokens_predicted", info.get("timings", {}).get("predicted_n")),
                "stop_type": info.get("stop_type"),
                "seconds": time.time() - t0,
            }
        except Exception as e:  # report every failure, a lost or errored request is what this checks for
            results[k] = {"stream": stream, "ok": False, "error": str(e), "seconds": time.time() - t0}

    threads = [threading.Thread(target=run, args=(k,)) for k in range(args.n_requests)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    all_ok = True
    for k, r in enumerate(results):
        all_ok = all_ok and r["ok"]
        if "error" in r:
            print(f"request {k} (stream={r['stream']}): ERROR after {r['seconds']:.1f}s: {r['error']}")
        else:
            status = "ok" if r["ok"] else f"MISMATCH ({r['mismatch']})"
            print(f"request {k} (stream={r['stream']}): {status}, {r['n_lines_ok']} consecutive lines, "
                  f"n_predicted = {r['n_predicted']}, stop = {r['stop_type']}, {r['seconds']:.1f}s")

    print("ALL OK" if all_ok else "FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
