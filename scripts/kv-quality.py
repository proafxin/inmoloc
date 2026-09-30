#!/usr/bin/env python3
# Long-context quality of a KV cache type against another, through the server as it is used.
#
#   run:     sends, for each document length, a document made of the repository's markdown with facts (needles) placed
#            at 10%, 50% and 90% of its depth, and two requests about it, one at a time and greedy:
#              qa      - the facts, asked at the end (accuracy)
#              summary - a summary of 256 tokens (a long greedy output to compare token by token)
#            and saves the text, the tokens and the top log-probabilities of every generated token
#   compare: a reference run (e.g. f16) against others (e.g. q8_0, and a second f16 run as the noise floor):
#            accuracy on the facts, the tokens the outputs agree on before they first differ, and the KL divergence of
#            the top log-probabilities over the tokens they agree on
#
# The documents depend only on the seed and the tokenizer of the server, so runs against different servers of the same
# model send the same prompts.

import argparse
import glob
import json
import math
import os
import random
import sys
import urllib.request

LENGTHS = [32000, 64000, 110000]
DEPTHS  = [0.1, 0.5, 0.9]
NAMES   = ["Aldebaran", "Borealis", "Cassiopeia", "Draconis", "Eridanus", "Fornax", "Hydrus", "Lyra", "Pavo"]


def post(url: str, path: str, body: dict) -> dict:
    req = urllib.request.Request(f"{url}{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=7200) as resp:
        return json.loads(resp.read())


def corpus(src: str) -> str:
    files = sorted(glob.glob(os.path.join(src, "docs", "**", "*.md"), recursive=True) +
                   glob.glob(os.path.join(src, "tools", "**", "*.md"), recursive=True) +
                   glob.glob(os.path.join(src, "examples", "**", "*.md"), recursive=True))
    return "\n\n".join(open(f, errors="replace").read() for f in files)


def documents(url: str, src: str, seed: int) -> list:
    text = corpus(src)
    n_text = len(post(url, "/tokenize", {"content": text})["tokens"])
    chars_per_token = len(text) / n_text

    rng  = random.Random(seed)
    docs = []
    for k, n_tokens in enumerate(LENGTHS):
        # each document starts elsewhere in the corpus, so that no two share a prefix
        n_chars = int(n_tokens * chars_per_token)
        start   = (k * len(text) // len(LENGTHS)) % len(text)
        body    = (text + "\n\n" + text)[start:start + n_chars]

        # the characters per token of this part of the corpus, so that the document has its length in tokens
        n_body  = len(post(url, "/tokenize", {"content": body})["tokens"])
        n_chars = int(n_chars * n_tokens / n_body)
        body    = (text + "\n\n" + text)[start:start + n_chars]

        names  = NAMES[3*k:3*k + 3]
        codes  = [str(rng.randint(100000, 999999)) for _ in names]
        needle = lambda name, code: f"\n\nThe archive code of the {name} project is {code}.\n\n"

        # at line breaks at the chosen depths, from the deepest so that earlier offsets stay valid
        for depth, name, code in sorted(zip(DEPTHS, names, codes), reverse=True):
            at = body.find("\n", int(depth * len(body)))
            at = at if at >= 0 else int(depth * len(body))
            body = body[:at] + needle(name, code) + body[at:]

        docs.append({"n_tokens": n_tokens, "body": body, "names": names, "codes": codes})
    return docs


def chat(url: str, prompt: str, max_tokens: int) -> dict:
    res = post(url, "/v1/chat/completions", {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "logprobs": True,
        "top_logprobs": 10,
        "chat_template_kwargs": {"enable_thinking": False},
    })
    choice = res["choices"][0]
    steps = []
    for c in (choice.get("logprobs") or {}).get("content") or []:
        steps.append({
            "token": c["token"],
            "logprob": c["logprob"],
            "top": {t["token"]: t["logprob"] for t in c.get("top_logprobs", [])},
        })
    return {"text": choice["message"]["content"], "steps": steps, "prompt_tokens": res.get("usage", {}).get("prompt_tokens")}


def run(args) -> None:
    out = {"label": args.label, "requests": []}
    for doc in documents(args.url, args.src, args.seed):
        head = "Read the following document.\n\n" + doc["body"] + "\n\n"
        qa = head + ("Using only the document above, what are the archive codes of the " + ", ".join(doc["names"]) +
                     " projects? Answer with one line per project, in the form 'Name: code'.")
        summary = head + "Summarize the document above in detail."

        for kind, prompt, n in (("qa", qa, 64), ("summary", summary, 256)):
            r = chat(args.url, prompt, n)
            r.update({"kind": kind, "n_tokens": doc["n_tokens"], "names": doc["names"], "codes": doc["codes"]})
            out["requests"].append(r)
            print(f"{args.label}: {kind:7s} {doc['n_tokens']:6d} tokens (prompt {r['prompt_tokens']}): {r['text'][:100]!r}", flush=True)

    with open(args.out, "w") as f:
        json.dump(out, f)


def kl_top(p: dict, q: dict) -> float:
    # KL(p || q) over the tokens either lists, renormalized; a token missing from a list gets that list's smallest
    # probability, an upper bound of what it could have
    keys = set(p) | set(q)
    pp = {k: math.exp(p[k]) if k in p else math.exp(min(p.values())) for k in keys}
    qq = {k: math.exp(q[k]) if k in q else math.exp(min(q.values())) for k in keys}
    sp, sq = sum(pp.values()), sum(qq.values())
    return sum(pp[k]/sp * math.log((pp[k]/sp) / (qq[k]/sq)) for k in keys)


def compare(args) -> None:
    runs = [json.load(open(f)) for f in args.runs]
    ref  = runs[0]

    print("accuracy on the facts (codes answered correctly):")
    for r in runs:
        line = []
        for req in r["requests"]:
            if req["kind"] != "qa":
                continue
            ok = [code in req["text"] and name in req["text"] for name, code in zip(req["names"], req["codes"])]
            line.append(f"{req['n_tokens']//1000}k {''.join('+' if o else '-' for o in ok)}")
        print(f"  {r['label']:10s} " + "  ".join(line) + "   (depth 10% 50% 90%)")

    print(f"\nagainst {ref['label']}: tokens equal before the first difference / generated, and over the equal tokens the KL "
          f"divergence of the top-10 log-probabilities (mean, max)")
    for r in runs[1:]:
        print(f"  {r['label']}:")
        for a, b in zip(ref["requests"], r["requests"]):
            sa, sb = a["steps"], b["steps"]
            n_eq = 0
            while n_eq < min(len(sa), len(sb)) and sa[n_eq]["token"] == sb[n_eq]["token"]:
                n_eq += 1
            kls = [kl_top(sa[i]["top"], sb[i]["top"]) for i in range(n_eq) if sa[i]["top"] and sb[i]["top"]]
            mean = sum(kls) / len(kls) if kls else float("nan")
            print(f"    {a['kind']:7s} {a['n_tokens']//1000:3d}k: {n_eq:3d} / {len(sa):3d} equal, KL mean {mean:.2e} max {max(kls, default=float('nan')):.2e}")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)
    r = sub.add_parser("run")
    r.add_argument("--url", default="http://localhost:8100")
    r.add_argument("--src", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--label", required=True)
    r.add_argument("--out", required=True)
    c = sub.add_parser("compare")
    c.add_argument("runs", nargs="+", help="result files, the first is the reference")
    args = p.parse_args()
    run(args) if args.mode == "run" else compare(args)


if __name__ == "__main__":
    main()
