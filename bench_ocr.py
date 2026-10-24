#!/usr/bin/env python3
"""OCR load test for an OpenAI-compatible server (llama.cpp, inmoloc, vLLM) serving PaddleOCR-VL.

The same pages go to every engine: PDF pages are rendered once with pdftoppm into a cache, chosen with
a fixed seed, so two runs with the same --pages/--seed/--dpi send the same images in the same order.
For each concurrency level it reports throughput, latency, failures, peak GPU memory and, if --container
is given, peak host RAM of that container. With --container the GPU memory is that of its processes only,
so several servers can stay loaded; run the load against one at a time, as they share the GPU compute. Outputs are saved so engines can be compared for agreement.

  python3 bench_ocr.py --url http://localhost:8110 --label llamacpp
  python3 bench_ocr.py --url http://localhost:8111 --label inmoloc
  python3 bench_ocr.py --compare bench_out/llamacpp.jsonl bench_out/inmoloc.jsonl

It waits for /health before sending anything; start it right after `docker compose up -d <service>` and the
"ready after" line is the startup time, including the few seconds it takes you to launch it.
Needs poppler-utils (pdftoppm, pdfinfo) and, for GPU memory, nvidia-smi.
"""

import argparse
import base64
import difflib
import json
import random
import re
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def page_count(pdf: Path) -> int:
    out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, timeout=30).stdout
    m = re.search(r"Pages:\s+(\d+)", out)
    return int(m.group(1)) if m else 0


def choose_pages(pdf_dir: Path, n: int, seed: int) -> list[tuple[Path, int]]:
    candidates = []
    for pdf in sorted(pdf_dir.rglob("*.pdf")):
        try:
            count = page_count(pdf)
        except Exception:  # noqa: BLE001
            continue
        candidates.extend((pdf, p) for p in range(1, count + 1))
    if not candidates:
        sys.exit(f"no PDF pages found in {pdf_dir}")
    rng = random.Random(seed)
    return rng.sample(candidates, min(n, len(candidates)))


def render(pages: list[tuple[Path, int]], dpi: int, cache: Path) -> list[Path]:
    cache.mkdir(parents=True, exist_ok=True)
    out = []
    for pdf, page in pages:
        stem = re.sub(r"[^A-Za-z0-9_.-]", "_", pdf.stem)[:60]
        target = cache / f"{stem}_p{page}_{dpi}.png"
        if not target.exists():
            prefix = cache / f"tmp_{stem}_{page}"
            subprocess.run(
                ["pdftoppm", "-r", str(dpi), "-png", "-singlefile", "-f", str(page), "-l", str(page), str(pdf), str(prefix)],
                check=True, capture_output=True, timeout=120,
            )
            (cache / f"{prefix.name}.png").rename(target)
        out.append(target)
    return out


def wait_ready(url: str, timeout: float) -> float:
    """Seconds until /health answers 200; the servers answer 503 while the model loads."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=2) as r:
                if r.status == 200:
                    return time.time() - start
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.25)
    sys.exit(f"{url} not ready after {timeout:.0f}s")


def ocr_request(url: str, model: str, image: Path, prompt: str, max_tokens: int) -> dict:
    b64 = base64.b64encode(image.read_bytes()).decode()
    body = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": prompt},
        ]}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    req = urllib.request.Request(
        f"{url}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            data = json.loads(r.read())
        usage = data.get("usage", {})
        return {
            "ok": True, "latency": time.time() - t0, "text": data["choices"][0]["message"]["content"],
            "completion_tokens": usage.get("completion_tokens", 0), "prompt_tokens": usage.get("prompt_tokens", 0),
        }
    except urllib.error.HTTPError as e:
        return {"ok": False, "latency": time.time() - t0, "error": f"HTTP {e.code}: {e.read()[:160].decode(errors='replace')}"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "latency": time.time() - t0, "error": f"{type(e).__name__}: {e}"}


class Sampler(threading.Thread):
    """Peak GPU memory (of the container's processes if --container is set) and host RAM of that container."""

    def __init__(self, container: str | None):
        super().__init__(daemon=True)
        self.container = container
        self.stop = threading.Event()
        self.gpu_peak = 0.0
        self.ram_peak = 0.0

    def container_pids(self) -> set[str]:
        out = subprocess.run(
            ["docker", "top", self.container, "-eo", "pid"], capture_output=True, text=True, timeout=10
        ).stdout.split()
        return set(out[1:])

    def gpu_used(self) -> float:
        """GPU memory in MiB: that of the container's processes if --container is set, else the whole GPU."""
        if not self.container:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip().splitlines()[0]
            return float(out)
        pids = self.container_pids()
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout
        total = 0.0
        for line in out.strip().splitlines():
            pid, mem = [x.strip() for x in line.split(",")]
            if pid in pids:
                total += float(mem)
        return total

    def run(self):
        while not self.stop.is_set():
            try:
                self.gpu_peak = max(self.gpu_peak, self.gpu_used())
            except Exception:  # noqa: BLE001
                pass
            if self.container:
                try:
                    out = subprocess.run(
                        ["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", self.container],
                        capture_output=True, text=True, timeout=10,
                    ).stdout.split("/")[0].strip()
                    m = re.match(r"([\d.]+)\s*([KMG]i?B)", out)
                    if m:
                        scale = {"KiB": 1 / 1024, "KB": 1 / 1024, "MiB": 1, "MB": 1, "GiB": 1024, "GB": 1024}[m.group(2)]
                        self.ram_peak = max(self.ram_peak, float(m.group(1)) * scale)
                except Exception:  # noqa: BLE001
                    pass
            self.stop.wait(1.0)


def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    return values[min(len(values) - 1, int(round(p / 100 * (len(values) - 1))))]


def run_level(args, images: list[Path], level: int, sink: list) -> dict:
    n = max(args.min_requests, level * args.per_slot)
    work = [images[i % len(images)] for i in range(n)]
    sampler = Sampler(args.container)
    sampler.start()
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=level) as pool:
        results = list(pool.map(lambda img: ocr_request(args.url, args.model, img, args.prompt, args.max_tokens), work))
    wall = time.time() - t0
    sampler.stop.set()
    ok = [r for r in results if r["ok"]]
    bad = [r for r in results if not r["ok"]]
    for img, r in zip(work, results):
        sink.append({"level": level, "page": img.name, **r})
    errors = sorted({r["error"][:90] for r in bad})
    return {
        "level": level, "requests": n, "ok": len(ok), "failed": len(bad), "wall_s": wall,
        "tok_s": sum(r["completion_tokens"] for r in ok) / wall if wall else 0,
        "req_s": len(ok) / wall if wall else 0,
        "lat_p50": pct([r["latency"] for r in ok], 50), "lat_p95": pct([r["latency"] for r in ok], 95),
        "gpu_peak_mib": sampler.gpu_peak, "ram_peak_mib": sampler.ram_peak, "errors": errors,
    }


def compare(a_path: str, b_path: str) -> None:
    def load(p):
        rows = {}
        for line in Path(p).read_text().splitlines():
            r = json.loads(line)
            if r.get("ok") and r["level"] == 1:
                rows.setdefault(r["page"], r["text"])
        return rows

    a, b = load(a_path), load(b_path)
    common = sorted(set(a) & set(b))
    if not common:
        sys.exit("no pages in common at concurrency 1")
    sims = [difflib.SequenceMatcher(None, a[p], b[p]).ratio() for p in common]
    exact = sum(a[p] == b[p] for p in common)
    print(f"{len(common)} pages in common: {exact} identical, mean similarity {statistics.mean(sims):.4f}, min {min(sims):.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8110")
    ap.add_argument("--label", default="run")
    ap.add_argument("--model", default="ocrt")
    ap.add_argument("--container", help="container name, to record its peak host RAM")
    ap.add_argument("--pdf-dir", default=str(Path.home() / "Downloads" / "ocr_input"))
    ap.add_argument("--cache-dir", default=str(Path.home() / ".cache" / "ocr_bench_pages"))
    ap.add_argument("--pages", type=int, default=64, help="distinct pages sampled")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dpi", type=int, default=144)
    ap.add_argument("--prompt", default="OCR:")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 8, 32, 128])
    ap.add_argument("--per-slot", type=int, default=2, help="requests per level = level x this")
    ap.add_argument("--min-requests", type=int, default=8)
    ap.add_argument("--out-dir", default="bench_out")
    ap.add_argument("--no-wait", action="store_true", help="do not wait for /health before sending requests")
    ap.add_argument("--prepare-only", action="store_true", help="render the pages and exit")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()

    if args.compare:
        compare(*args.compare)
        return

    pages = choose_pages(Path(args.pdf_dir), args.pages, args.seed)
    images = render(pages, args.dpi, Path(args.cache_dir))
    sizes = [p.stat().st_size for p in images]
    print(f"{len(images)} pages rendered at {args.dpi} dpi, {statistics.median(sizes) / 1024:.0f} KiB median PNG")
    if args.prepare_only:
        return

    if not args.no_wait:
        print(f"ready after {wait_ready(args.url, 600):.1f}s")

    for img in images[:2]:  # warm-up, not counted
        r = ocr_request(args.url, args.model, img, args.prompt, args.max_tokens)
        if not r["ok"]:
            sys.exit(f"warm-up request failed, not starting the load: {r['error']}")

    rows, sink = [], []
    print(f"{'level':>5} {'reqs':>5} {'ok':>4} {'fail':>4} {'wall s':>7} {'tok/s':>7} {'req/s':>6} "
          f"{'p50 s':>6} {'p95 s':>6} {'GPU MiB':>8} {'RAM MiB':>8}")
    for level in args.levels:
        r = run_level(args, images, level, sink)
        rows.append(r)
        print(f"{r['level']:>5} {r['requests']:>5} {r['ok']:>4} {r['failed']:>4} {r['wall_s']:>7.1f} {r['tok_s']:>7.0f} "
              f"{r['req_s']:>6.2f} {r['lat_p50']:>6.1f} {r['lat_p95']:>6.1f} {r['gpu_peak_mib']:>8.0f} {r['ram_peak_mib']:>8.0f}")
        for e in r["errors"]:
            print(f"        error: {e}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.label}.jsonl").write_text("\n".join(json.dumps(x) for x in sink))
    (out / f"{args.label}_summary.json").write_text(json.dumps({"args": vars(args), "levels": rows}, indent=2))
    print(f"saved {out / (args.label + '.jsonl')}")


if __name__ == "__main__":
    main()
