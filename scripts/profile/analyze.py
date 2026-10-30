#!/usr/bin/env python3
# Summarizes Nsight Systems traces written by profile-server.sh: GPU busy time and idle gaps, kernel time by category,
# and the CUDA API calls the CPU spends its time in.
#
#   python3 scripts/profile/analyze.py ~/Projects/bench-results/profile/<name>.sqlite [more.sqlite ...]

import collections
import sqlite3
import sys


def category(name: str) -> str:
    rules = [
        ("mmq",       ("mul_mat_q<",)),
        ("mmvq",      ("mul_mat_vec_q",)),
        ("mmvf",      ("mul_mat_vec_f",)),
        ("attention", ("flash_attn",)),
        ("gdn",       ("gated_delta_net", "gdn_single")),
        ("ssm_conv",  ("ssm_conv",)),
        ("get/set rows", ("get_rows", "set_rows", "k_set_rows", "k_get_rows")),
        ("copy/cont", ("cpy", "concat", "dequantize", "convert")),
        ("norm",      ("norm",)),
        ("elementwise", ("unary", "binbcast", "k_bin_bcast", "scale", "silu", "add", "mul", "sigmoid", "softplus", "exp")),
        ("softmax/top-k", ("soft_max", "argsort", "top_k", "topk")),
    ]
    for cat, keys in rules:
        if any(k in name for k in keys):
            return cat
    return "other"


def analyze(path: str) -> None:
    db = sqlite3.connect(path)
    tables = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
    acts = [t for t in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET") if t in tables]
    rows = db.execute(" union all ".join(f"select start, end from {t}" for t in acts) + " order by start").fetchall()

    t0, t1 = rows[0][0], max(r[1] for r in rows)
    busy = 0
    gaps = []
    cs, ce = rows[0]
    for s, e in rows[1:]:
        if s > ce:
            busy += ce - cs
            gaps.append(s - ce)
            cs, ce = s, e
        else:
            ce = max(ce, e)
    busy += ce - cs
    span = t1 - t0

    print(f"== {path}")
    print(f"   span {span/1e9:.2f} s, GPU busy {busy/1e9:.2f} s ({100*busy/span:.1f}%), idle {(span-busy)/1e9:.2f} s")
    for lo, hi in [(0, 1e4), (1e4, 1e5), (1e5, 1e6), (1e6, 1e7), (1e7, 1e12)]:
        k = [g for g in gaps if lo <= g < hi]
        print(f"   idle gaps {lo/1e3:>6.0f}-{hi/1e3:<9.0f} us: {len(k):7d}  total {sum(k)/1e9:6.2f} s")

    cat_t = collections.Counter()
    cat_n = collections.Counter()
    for name, t, n in db.execute("""select s.value, sum(k.end - k.start), count(*) from CUPTI_ACTIVITY_KIND_KERNEL k
                                    join StringIds s on s.id = k.demangledName group by 1"""):
        c = category(name)
        cat_t[c] += t
        cat_n[c] += n
    total = sum(cat_t.values())
    print(f"   kernel time {total/1e9:.2f} s in {sum(cat_n.values())} kernels")
    for c, t in cat_t.most_common():
        print(f"     {c:14s} {t/1e9:6.3f} s  {100*t/total:5.1f}%  {cat_n[c]:7d} kernels  {t/max(cat_n[c],1)/1e3:7.1f} us avg")

    if "CUPTI_ACTIVITY_KIND_RUNTIME" in tables:
        print("   CUDA API time (CPU side)")
        q = """select s.value, sum(r.end - r.start), count(*) from CUPTI_ACTIVITY_KIND_RUNTIME r
               join StringIds s on s.id = r.nameId group by 1 order by 2 desc limit 6"""
        for name, t, n in db.execute(q):
            print(f"     {name[:40]:40s} {t/1e9:6.2f} s  {n:7d} calls")


def window_tokens(log_path: str, seconds: float = 20.0) -> None:
    # tokens generated and prompt tokens processed in the last `seconds` of the server log (the profiled window),
    # interpolated from the per-request progress lines
    import re

    def ts(s: str) -> float:
        m, sec, ms, us = s.split(".")
        return int(m)*60 + int(sec) + int(ms)/1e3 + int(us)/1e6

    gen = collections.defaultdict(list)
    pp = {}
    start = {}
    last = 0.0
    for line in open(log_path, errors="replace"):
        m = re.match(r"(\d+\.\d+\.\d+\.\d+) .*task (\d+) \| (.*)", line)
        if not m:
            continue
        t, task, rest = ts(m[1]), int(m[2]), m[3]
        last = t
        if "processing task" in rest:
            start[task] = t
        if (g := re.search(r"n_gen = +(\d+)", rest)):
            gen[task].append((t, int(g[1])))
        if (g := re.search(r"\| +eval time = .*/ +(\d+) tokens", " | " + rest)) and "prompt" not in rest:
            gen[task].append((t, int(g[1])))
        if (g := re.search(r"prompt eval time = +([\d.]+) ms / +(\d+) tokens", rest)) and task in start:
            pp[task] = [(start[task], 0), (start[task] + float(g[1])/1e3, int(g[2]))]

    t1 = last
    t0 = t1 - seconds

    def at(pts, t):
        pts = sorted(pts)
        prev = (pts[0][0], 0)
        for p in pts:
            if p[0] >= t:
                return p[1] if p[0] == prev[0] else prev[1] + (p[1]-prev[1]) * (t-prev[0]) / (p[0]-prev[0])
            prev = p
        return pts[-1][1]

    n_gen = sum(at(v, t1) - at(v, t0) for v in gen.values())
    n_pp  = sum(at(v, t1) - at(v, t0) for v in pp.values())
    print(f"   window tokens: generated {n_gen:.0f} ({n_gen/seconds:.1f}/s), prompt {n_pp:.0f} ({n_pp/seconds:.0f}/s)")


for p in sys.argv[1:]:
    analyze(p)
    log = p.replace(".sqlite", "-server.log")
    try:
        window_tokens(log)
    except (OSError, IndexError):
        pass
