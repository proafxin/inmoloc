#!/usr/bin/env python3
# Benchmark suite: the same models, settings and workloads on several servers (llama.cpp and inmoloc images), so that a
# change counts as a gain only when it shows across workloads and beyond the run-to-run spread.
#
# Tiers, each run per model as listed in the config (scripts/bench-suite.json):
#   engine - `bench` (prompt and generation speed of one sequence) and `batched-bench` (decoding of several sequences at
#            once) of the unified binary: kernels and batching, no server and no scheduler
#   server - scripts/bench-server-concurrency.py against the server, for each workload (prompt and output sizes, fixed by
#            the seed so that every server gets the same requests) at several concurrency levels
#   embed  - embedding requests of random lengths at several concurrency levels
#   vision - chat requests with one image each (only for models with a vision encoder), from a directory of images
#
# Every repeat runs all the targets one after another (A B C, A B C, ...), so that a slow period of the machine does not
# land on one target only; the summary reports the median and the range over the repeats.
#
# The servers are the bench-* services of compose.yaml. Run from the repository root, with nothing else on the GPU:
#   python3 scripts/bench-suite.py --out ~/Projects/bench-results/run1                  (default models, all targets)
#   python3 scripts/bench-suite.py --models qwen3.8-mtp --tiers engine --targets base current
#   python3 scripts/bench-suite.py --models qwen3.8-mtp --tiers server --workloads short long-in long-out mixed
#   python3 scripts/bench-suite.py --summarize ~/Projects/bench-results/run1     (rebuild the summary from the raw files)

import argparse
import base64
import glob
import json
import os
import random
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PORT = {"llamacpp": 8130, "inmoloc": 8131}

WORDS = ("river stone market winter engine signal harbor letter garden copper season ledger bridge lantern "
         "orchard mirror valley thunder canvas anchor meadow compass silver timber beacon quarry").split()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def compose_env(target: dict, model: dict | None = None) -> dict:
    env = dict(os.environ)
    env["LLAMACPP_IMAGE" if target["kind"] == "llamacpp" else "INMOLOC_IMAGE"] = target["image"]
    if model is not None:
        env["BENCH_MODEL"]         = model["file"]
        env["BENCH_COMMON_ARGS"]   = model.get("common_args", "")
        env["BENCH_LLAMACPP_ARGS"] = model.get("llamacpp_args", "")
        env["BENCH_INMOLOC_ARGS"]  = model.get("inmoloc_args", "")
    return env


def compose(args: list, env: dict, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", "compose", "--profile", "bench", *args], cwd=ROOT, env=env, **kw)


def jsonl(text: str) -> list:
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


#
# engine tier
#

def run_engine(model: dict, target: dict, raw: str, ecfg: dict) -> dict:
    env = compose_env(target)
    service = f"bench-engine-{target['kind']}"
    path = f"/models/{model['file']}"
    extra = model.get("engine_args", "").split()
    res = {}

    b = ecfg.get("bench", {})
    cmd = ["run", "--rm", service, "bench", "-m", path, "-ngl", "999", "-r", str(b.get("repetitions", 3)),
           "-p", b.get("n_prompt", "512"), "-n", b.get("n_gen", "128"), "-o", "jsonl", *extra]
    out = compose(cmd, env, capture_output=True, text=True)
    open(raw + "-bench.log", "w").write(out.stdout + "\n--- stderr ---\n" + out.stderr)
    for row in jsonl(out.stdout):
        test = f"pp{row['n_prompt']}" if row.get("n_prompt") else f"tg{row['n_gen']}"
        res[f"bench/{test}/tok_s"] = row["avg_ts"]

    bb = ecfg.get("batched", {})
    cmd = ["run", "--rm", service, "batched-bench", "-m", path, "-ngl", "999", "-c", str(model.get("engine_ctx", 20480)),
           "-npp", bb.get("npp", "512"), "-ntg", bb.get("ntg", "64"), "-npl", bb.get("npl", "1,8,32"),
           "--output-format", "jsonl", *extra]
    out = compose(cmd, env, capture_output=True, text=True)
    open(raw + "-batched.log", "w").write(out.stdout + "\n--- stderr ---\n" + out.stderr)
    for row in jsonl(out.stdout):
        key = f"batched/pp{row['pp']}-tg{row['tg']}-pl{row['pl']}"
        res[key + "/pp_tok_s"] = row["speed_pp"]
        res[key + "/tg_tok_s"] = row["speed_tg"]

    if not res:
        log(f"  no engine results, see {raw}-*.log")
    return res


#
# server tiers
#

def start_server(model: dict, target: dict, raw: str) -> str | None:
    env = compose_env(target, model)
    service = f"bench-{target['kind']}"
    compose(["rm", "-sf", service], env, capture_output=True)
    if compose(["up", "-d", service], env).returncode != 0:
        return None
    url = f"http://localhost:{PORT[target['kind']]}"
    t0 = time.time()
    while time.time() - t0 < 1800:
        try:
            with urllib.request.urlopen(url + "/health", timeout=5) as r:
                if r.status == 200:
                    log(f"  {service} ready in {time.time() - t0:.0f} s")
                    return url
        except Exception:  # noqa: BLE001
            pass
        state = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", service], capture_output=True, text=True)
        if state.stdout.strip() != "true":
            log(f"  {service} exited before it was ready, see {raw}-server.log")
            stop_server(target, raw)
            return None
        time.sleep(5)
    log(f"  {service} not ready after 30 min")
    stop_server(target, raw)
    return None


def stop_server(target: dict, raw: str) -> None:
    service = f"bench-{target['kind']}"
    logs = subprocess.run(["docker", "logs", service], capture_output=True, text=True)
    open(raw + "-server.log", "w").write(logs.stdout + logs.stderr)
    compose(["rm", "-sf", service], compose_env(target), capture_output=True)


def run_server(model: dict, target: dict, raw: str, workloads: dict, seed: int) -> dict:
    url = start_server(model, target, raw)
    if url is None:
        return {}
    res = {}
    try:
        for name in model["workloads"]:
            w = workloads[name]
            out_json = f"{raw}-{name}.json"
            cmd = [sys.executable, os.path.join(ROOT, "scripts", "bench-server-concurrency.py"), "--url", url,
                   "--levels", *map(str, model["levels"]), "--seed", str(seed), "--json", out_json,
                   "--prompt-tokens", str(w["prompt_tokens"]), "--max-tokens", str(w["max_tokens"]),
                   "--prompt-tokens-max", str(w.get("prompt_tokens_max", 0)),
                   "--max-tokens-max", str(w.get("max_tokens_max", 0)),
                   "--per-client", str(w.get("per_client", 1))]
            log(f"  workload {name}")
            out = subprocess.run(cmd, capture_output=True, text=True)
            open(f"{raw}-{name}.log", "w").write(out.stdout + out.stderr)
            print(out.stdout, flush=True)
            try:
                with urllib.request.urlopen(url + "/metrics", timeout=30) as r:
                    open(f"{raw}-{name}-metrics.txt", "wb").write(r.read())
            except Exception as e:  # noqa: BLE001
                log(f"  metrics not saved: {e}")
            if os.path.exists(out_json):
                for row in json.load(open(out_json)):
                    key = f"{name}/c{row['level']}"
                    for m in ("agg_tps", "ttft_s", "itl_p50_ms", "itl_p99_ms", "accept", "failed"):
                        if row.get(m) is not None:
                            res[f"{key}/{m}"] = row[m]
    finally:
        stop_server(target, raw)
    return res


def post_json(url: str, body: dict, timeout: float = 3600) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def run_pool(fn, items: list, conc: int) -> tuple[float, list]:
    results = [None] * len(items)
    next_i = [0]
    lock = threading.Lock()

    def worker():
        while True:
            with lock:
                i = next_i[0]
                next_i[0] += 1
            if i >= len(items):
                return
            try:
                results[i] = fn(items[i])
            except Exception as e:  # noqa: BLE001
                results[i] = {"error": str(e)}

    threads = [threading.Thread(target=worker) for _ in range(conc)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return time.time() - t0, results


def run_embed(model: dict, target: dict, raw: str, seed: int) -> dict:
    url = start_server(model, target, raw)
    if url is None:
        return {}
    cfg = model["embed"]
    rng = random.Random(seed)
    texts = [" ".join(rng.choice(WORDS) for _ in range(rng.randint(cfg["min_words"], cfg["max_words"])))
             for _ in range(cfg["n_requests"])]
    res = {}
    lines = []
    try:
        def one(text):
            t0 = time.time()
            r = post_json(url + "/v1/embeddings", {"input": text})
            return {"latency": time.time() - t0, "tokens": r.get("usage", {}).get("prompt_tokens", 0)}

        for conc in model["levels"]:
            wall, outs = run_pool(one, texts, conc)
            good = [o for o in outs if o and "error" not in o]
            lat = sorted(o["latency"] for o in good)
            res[f"embed/c{conc}/req_s"] = len(good) / wall
            res[f"embed/c{conc}/tok_s"] = sum(o["tokens"] for o in good) / wall
            res[f"embed/c{conc}/p50_ms"] = lat[len(lat) // 2] * 1e3 if lat else float("nan")
            res[f"embed/c{conc}/failed"] = len(outs) - len(good)
            lines.append(f"c{conc}: {res[f'embed/c{conc}/req_s']:.1f} req/s, {res[f'embed/c{conc}/tok_s']:.0f} tok/s, "
                         f"failed {len(outs) - len(good)}")
            log("  " + lines[-1])
    finally:
        open(raw + "-embed.log", "w").write("\n".join(lines) + "\n")
        stop_server(target, raw)
    return res


def run_vision(model: dict, target: dict, raw: str, seed: int) -> dict:
    cfg = model["vision"]
    images = sorted(f for ext in ("png", "jpg", "jpeg", "webp")
                    for f in glob.glob(os.path.join(os.path.expanduser(cfg["images"]), f"**/*.{ext}"), recursive=True))
    if not cfg.get("images") or not images:
        log("  vision: no images, set models[].vision.images in the config")
        return {}
    url = start_server(model, target, raw)
    if url is None:
        return {}
    rng = random.Random(seed)
    picks = [rng.choice(images) for _ in range(cfg["n_requests"])]
    res = {}
    lines = []
    try:
        def one(path):
            ext = os.path.splitext(path)[1].lstrip(".").replace("jpg", "jpeg")
            data = base64.b64encode(open(path, "rb").read()).decode()
            body = {
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/{ext};base64,{data}"}},
                    {"type": "text", "text": cfg["prompt"]}]}],
                "max_tokens": cfg["max_tokens"], "temperature": 0.0,
            }
            t0 = time.time()
            r = post_json(url + "/v1/chat/completions", body)
            return {"latency": time.time() - t0, "tokens": r.get("usage", {}).get("completion_tokens", 0)}

        for conc in model["levels"]:
            wall, outs = run_pool(one, picks, conc)
            good = [o for o in outs if o and "error" not in o]
            lat = sorted(o["latency"] for o in good)
            res[f"vision/c{conc}/tok_s"] = sum(o["tokens"] for o in good) / wall
            res[f"vision/c{conc}/req_s"] = len(good) / wall
            res[f"vision/c{conc}/p50_s"] = lat[len(lat) // 2] if lat else float("nan")
            res[f"vision/c{conc}/failed"] = len(outs) - len(good)
            lines.append(f"c{conc}: {res[f'vision/c{conc}/tok_s']:.0f} tok/s, {res[f'vision/c{conc}/req_s']:.2f} req/s, "
                         f"failed {len(outs) - len(good)}")
            log("  " + lines[-1])
    finally:
        open(raw + "-vision.log", "w").write("\n".join(lines) + "\n")
        stop_server(target, raw)
    return res


#
# summary
#

# metrics where lower is better
LOWER_BETTER = ("ttft_s", "itl_p50_ms", "itl_p99_ms", "p50_ms", "p50_s", "failed")


def summarize(out: str) -> None:
    results = [json.loads(line) for line in open(os.path.join(out, "results.jsonl")) if line.strip()]
    run_cfg = json.load(open(os.path.join(out, "run.json"))).get("config", {}) if os.path.exists(os.path.join(out, "run.json")) else {}

    # model -> tier -> metric -> target -> values over the repeats
    table = {}
    targets = []
    for r in results:
        if r["target"] not in targets:
            targets.append(r["target"])
        for metric, value in r["metrics"].items():
            table.setdefault(r["model"], {}).setdefault(r["tier"], {}).setdefault(metric, {}).setdefault(r["target"], []).append(value)

    def cell(values):
        values = [v for v in values if v is not None and v == v]
        if not values:
            return "-"
        med = statistics.median(values)
        if len(values) == 1:
            return f"{med:.4g}"
        return f"{med:.4g} [{min(values):.4g}-{max(values):.4g}]"

    # the comparisons: [target, reference] pairs of the config, else every target against the first one
    pairs = [p for p in run_cfg.get("compare", []) if p[0] in targets and p[1] in targets] or \
            [[t, targets[0]] for t in targets[1:]]

    lines = ["# Benchmark summary", "",
             "Median over the repeats, the range in brackets. The last columns compare a target with a reference: a",
             "difference is marked when the two ranges do not overlap (+ better, - worse).", ""]
    summary = {}
    for model, tiers in table.items():
        for tier, metrics in tiers.items():
            lines += [f"## {model}: {tier}", ""]
            lines.append("| metric | " + " | ".join(targets) + "".join(f" | {t} vs {r}" for t, r in pairs) + " |")
            lines.append("| --- " * (1 + len(targets) + len(pairs)) + "|")
            for metric in sorted(metrics):
                vals = metrics[metric]
                row = [metric] + [cell(vals.get(t, [])) for t in targets]
                for t, r in pairs:
                    ref = [v for v in vals.get(r, []) if v is not None and v == v]
                    cur = [v for v in vals.get(t, []) if v is not None and v == v]
                    if not ref or not cur or statistics.median(ref) == 0:
                        row.append("-")
                        continue
                    ratio = statistics.median(cur) / statistics.median(ref) - 1
                    lower = metric.endswith(LOWER_BETTER)
                    separate = min(cur) > max(ref) or max(cur) < min(ref)
                    better = (statistics.median(cur) < statistics.median(ref)) == lower
                    mark = (" +" if better else " -") if separate and len(ref) > 1 and len(cur) > 1 else ""
                    row.append(f"{100 * ratio:+.1f}%{mark}")
                lines.append("| " + " | ".join(row) + " |")
                summary.setdefault(model, {}).setdefault(tier, {})[metric] = {t: vals.get(t, []) for t in targets}
            lines.append("")

    open(os.path.join(out, "summary.md"), "w").write("\n".join(lines))
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
    log(f"summary: {os.path.join(out, 'summary.md')}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(ROOT, "scripts", "bench-suite.json"))
    parser.add_argument("--out", default=os.path.expanduser(f"~/Projects/bench-results/{time.strftime('%Y%m%d-%H%M%S')}"))
    parser.add_argument("--models", nargs="*", help="model names of the config to run (default: all)")
    parser.add_argument("--tiers", nargs="*", help="tiers to run: engine server embed vision (default: those of each model)")
    parser.add_argument("--targets", nargs="*", help="target names of the config to run (default: all)")
    parser.add_argument("--repeats", type=int, help="overrides the repeats of the config")
    parser.add_argument("--workloads", nargs="*", help="server workloads to run, overrides those of each model")
    parser.add_argument("--levels", type=int, nargs="*", help="concurrency levels, override those of each model")
    parser.add_argument("--summarize", metavar="DIR", help="only rebuild the summary of an earlier run")
    args = parser.parse_args()

    if args.summarize:
        summarize(args.summarize)
        return 0

    cfg = json.load(open(args.config))
    targets = [t for t in cfg["targets"] if not args.targets or t["name"] in args.targets]
    # models marked "default": false run only when named
    models = [m for m in cfg["models"] if (m["name"] in args.models if args.models else m.get("default", True))]
    for m in models:
        if args.workloads:
            m["workloads"] = args.workloads
        if args.levels:
            m["levels"] = args.levels
    repeats = args.repeats or cfg.get("repeats", 3)

    os.makedirs(os.path.join(args.out, "raw"), exist_ok=True)
    json.dump({"config": cfg, "targets": [t["name"] for t in targets], "models": [m["name"] for m in models],
               "repeats": repeats, "started": time.strftime("%Y-%m-%d %H:%M:%S")},
              open(os.path.join(args.out, "run.json"), "w"), indent=1)
    results_path = os.path.join(args.out, "results.jsonl")

    for model in models:
        tiers = [t for t in model["tiers"] if not args.tiers or t in args.tiers]
        for tier in tiers:
            for rep in range(repeats):
                for target in targets:
                    log(f"{model['name']} / {tier} / {target['name']} ({target['image']}) / repeat {rep + 1} of {repeats}")
                    raw = os.path.join(args.out, "raw", f"{model['name']}-{tier}-{target['name']}-r{rep + 1}")
                    if tier == "engine":
                        metrics = run_engine(model, target, raw, cfg.get("engine", {}))
                    elif tier == "server":
                        metrics = run_server(model, target, raw, cfg["workloads"], seed=rep)
                    elif tier == "embed":
                        metrics = run_embed(model, target, raw, seed=rep)
                    elif tier == "vision":
                        metrics = run_vision(model, target, raw, seed=rep)
                    else:
                        log(f"  unknown tier {tier}")
                        continue
                    with open(results_path, "a") as f:
                        f.write(json.dumps({"model": model["name"], "tier": tier, "target": target["name"],
                                            "repeat": rep + 1, "metrics": metrics}) + "\n")
            summarize(args.out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
