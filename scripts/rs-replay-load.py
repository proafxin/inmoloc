#!/usr/bin/env python3
# Sends waves of concurrent greedy requests of different lengths and prints each output on one line, in a fixed
# order, so two servers can be compared with diff. Natural language keeps the draft acceptance low, so the
# recurrent state is rolled back often; the short requests finish first and leave idle cells between the long
# ones; the second wave reuses the slots of the first.

import argparse
import json
import threading
import urllib.request

WAVES = [
    [("Describe a lighthouse on a stormy winter night.", 48),
     ("Name two colors.", 8),
     ("Explain how bread rises, in plain words.", 48),
     ("Say hello in French.", 6)],
    [("Write two sentences about the sea.", 32),
     ("Give one reason people keep bonsai trees.", 32),
     ("What is a B-tree used for?", 32),
     ("Describe the smell of rain.", 32)],
]


def ask(url: str, prompt: str, n: int, out: list, i: int) -> None:
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": n,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=3600) as resp:
            out[i] = json.loads(resp.read())["choices"][0]["message"]["content"].replace("\n", " ")
    except Exception as e:  # noqa: BLE001
        out[i] = f"ERROR {e}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8100")
    args = p.parse_args()

    for w, wave in enumerate(WAVES):
        out = [None] * len(wave)
        threads = [threading.Thread(target=ask, args=(args.url, q, n, out, i)) for i, (q, n) in enumerate(wave)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for i, text in enumerate(out):
            print(f"wave {w} req {i}: {text}")


if __name__ == "__main__":
    main()
