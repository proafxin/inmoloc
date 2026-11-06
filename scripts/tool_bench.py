#!/usr/bin/env uv run
"""
Simplistic tool call benchmarks for local-inference-server and ollama.

Essentially runs the tests at tests/server/unit/test_tool_call.py N times, at different temperatures and on different backends (current local-inference-server, baseline local-inference-server and ollama),
and plots the results of multiple runs (from same .jsonl file or multiple ones) as a success rate heatmap.

Simple usage example:

    cmake -B build && cmake --build build --config Release -j -t local-inference-server

    export LOCAL_INFERENCE_SERVER_BIN_PATH=$PWD/build/bin/local-inference-server
    export LOCAL_INFERENCE_CACHE=${LOCAL_INFERENCE_CACHE:-$HOME/Library/Caches/llama.cpp}

    ./scripts/tool_bench.py run --n 10 --temp -1 --temp 0 --temp 1 --temp 2 --temp 5 --llama-baseline $PWD/buildMaster/bin/local-inference-server --output qwen14b.jsonl --hf bartowski/Qwen2.5-14B-Instruct-GGUF:Q4_K_L
    ./scripts/tool_bench.py run --n 30 --temp -1 --temp 0 --temp 1 --model "Qwen 2.5 1.5B Q4_K_M"      --output qwen1.5b.jsonl  --hf bartowski/Qwen2.5-1.5B-Instruct-GGUF      --ollama qwen2.5:1.5b-instruct-q4_K_M
    ./scripts/tool_bench.py run --n 30 --temp -1 --temp 0 --temp 1 --model "Qwen 2.5 Coder 7B Q4_K_M"  --output qwenc7b.jsonl   --hf bartowski/Qwen2.5-Coder-7B-Instruct-GGUF  --ollama qwen2.5-coder:7b

    ./scripts/tool_bench.py plot *.jsonl                         # Opens window w/ heatmap
    ./scripts/tool_bench.py plot qwen*.jsonl  --output qwen.png  # Saves heatmap to qwen.png

(please see ./scripts/tool_bench.sh for a more complete example)
"""

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pytest",
#     "pandas",
#     "matplotlib",
#     "seaborn",
#     "requests",
#     "wget",
#     "typer",
# ]
# ///
import atexit
import json
import logging
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from statistics import mean, median
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer

sys.path.insert(0, Path(__file__).parent.parent.as_posix())
if True:
    from tools.server.tests.unit.test_tool_call import do_test_calc_result, do_test_hello_world, do_test_weather  # ty: ignore[unresolved-import]
    from tools.server.tests.utils import ServerProcess  # ty: ignore[unresolved-import]


@contextmanager
def scoped_server(sp: ServerProcess):
    def stop():
        nonlocal sp
        if sp is not None:
            sp.stop()
            sp = None  # type: ignore

    atexit.register(stop)
    yield sp
    stop()


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = typer.Typer()


@app.command()
def plot(
    files: list[Path], output: Path | None = None, test_regex: str | None = None, server_regex: str | None = None
) -> None:

    lines: list[dict] = []
    for file in files:
        if not file.exists():
            logger.error("File not found: %s", file)
            continue

        try:
            with file.open() as f:
                raw_data = f.read()
            logger.info(f"Reading {file} ({len(raw_data)} bytes)")

            for line_num, line in enumerate(raw_data.split("\n"), 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    lines.append(record)
                except json.JSONDecodeError as e:
                    logger.warning("Invalid JSON at %s:%s - %s", file, line_num, e)
        except Exception as e:
            logger.exception("Error processing %s: %s", file, e)

    if not lines:
        raise Exception("No valid data was loaded")

    data_dict: dict[tuple, float] = {}
    models: list[str] = []
    temps = set()
    tests = set()
    server_names = set()
    total_counts = set()
    for rec in lines:
        try:
            model = rec["model"]
            temp = rec["temp"]
            server_name = rec["server_name"]
            test = rec["test"]
            success = rec["success_ratio"]
            success_count = rec["success_count"]
            failure_count = rec["failure_count"]
            total_count = success_count + failure_count
            total_counts.add(total_count)

            if test_regex and not re.search(test_regex, test):
                continue

            if server_regex and not re.search(server_regex, server_name):
                continue

            data_dict[model, temp, server_name, test] = success

            if model not in models:
                models.append(model)
            temps.add(temp)
            tests.add(test)
            server_names.add(server_name)

        except KeyError as e:
            logger.warning("Missing required field in record: %s", e)

    if len(total_counts) > 1:
        logger.warning("Total counts are not consistent: %s", total_counts)

    # Sort the collected values
    temps = sorted(temps, key=lambda x: x if x is not None else -1)
    tests = sorted(tests)
    server_names = sorted(server_names)

    logger.info(f"Processed {len(lines)} lines")
    logger.info(f"Found {len(data_dict)} valid data points")
    logger.info("Models: %s", models)
    logger.info("Temperatures: %s", temps)
    logger.info("Tests: %s", tests)
    logger.info("Servers: %s", server_names)

    matrix: list[list[float]] = []
    index: list[str] = []

    all_cols = [(server_name, test) for server_name in server_names for test in tests]
    for model in models:
        for temp in temps:
            index.append(f"{model} @ {temp}")
            row_vals = [data_dict.get((model, temp, server_name, test), np.nan) for server_name, test in all_cols]
            matrix.append(row_vals)

    columns: list[str] = [f"{server_name}\n{test}" for server_name, test in all_cols]

    df = pd.DataFrame(matrix, index=np.array(index), columns=np.array(columns))

    plt.figure(figsize=(12, 6))

    sns.heatmap(
        df,
        annot=True,
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        cbar=True,
        fmt=".2f",
        center=0.5,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Success Ratio"},
    )

    plt.title(
        f"Tool Call Bench (n = {str(min(total_counts)) if len(total_counts) == 1 else f'{min(total_counts)}-{max(total_counts)}'})\nSuccess Ratios by Server & Test",
        pad=20,
    )
    plt.xlabel("Server & Test", labelpad=10)
    plt.ylabel("Model @ Temperature", labelpad=10)

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=300, bbox_inches="tight")
        logger.info("Plot saved to %s", output)
    else:
        plt.show()


@app.command()
def run(
    output: Annotated[Path, typer.Option(help="Output JSON file")],
    model: Annotated[str | None, typer.Option(help="Name of the model to test (server agnostic)")] = None,
    hf: Annotated[
        str | None,
        typer.Option(help="GGUF huggingface model repo id (+ optional quant) to test w/ local-inference-server"),
    ] = None,
    chat_template: Annotated[str | None, typer.Option(help="Chat template override for local-inference-server")] = None,
    chat_template_file: Annotated[
        str | None, typer.Option(help="Chat template file override for local-inference-server")
    ] = None,
    ollama: Annotated[str | None, typer.Option(help="Ollama model tag to test")] = None,
    llama_baseline: Annotated[
        str | None, typer.Option(help="local-inference-server baseline binary path to use as baseline")
    ] = None,
    n: Annotated[int, typer.Option(help="Number of times to run each test")] = 10,
    temp: Annotated[list[float] | None, typer.Option(help="Set of temperatures to test")] = None,
    top_p: Annotated[float | None, typer.Option(help="top_p")] = None,
    top_k: Annotated[int | None, typer.Option(help="top_k")] = None,
    ctk: Annotated[str | None, typer.Option(help="ctk")] = None,
    ctv: Annotated[str | None, typer.Option(help="ctv")] = None,
    fa: Annotated[bool | None, typer.Option(help="fa")] = None,
    seed: Annotated[int | None, typer.Option(help="Random seed")] = None,
    port: Annotated[int, typer.Option(help="local-inference-server port")] = 8084,
    force: Annotated[bool, typer.Option(help="Force overwrite of output file")] = False,
    append: Annotated[bool, typer.Option(help="Append to output file")] = False,
    test_hello_world: Annotated[bool, typer.Option(help="Whether to run the hello world test")] = True,
    test_weather: Annotated[bool, typer.Option(help="Whether to run the weather test")] = True,
    test_calc_result: Annotated[bool, typer.Option(help="Whether to run the calc result test")] = False,
) -> None:
    # Check only one of output and append

    n_predict = 512  # High because of DeepSeek R1
    # n_ctx = 8192
    n_ctx = 2048

    if model is None:
        if hf is not None:
            model = hf.split("/")[-1]
        elif ollama is not None:
            model = ollama

    assert force or append or not output.exists(), f"Output file already exists: {output}; use --force to overwrite"

    with output.open("a" if append else "w") as output_file:

        def run(
            server: ServerProcess,
            *,
            server_name: str,
            model_id: str,
            temp: float | None = None,
            output_kwargs=None,
            request_kwargs=None,
        ):
            if request_kwargs is None:
                request_kwargs = {}
            if output_kwargs is None:
                output_kwargs = {}
            request_kwargs = {**request_kwargs}
            if temp is not None:
                request_kwargs["temperature"] = temp
            if top_p is not None:
                request_kwargs["top_p"] = top_p
            if top_k is not None:
                request_kwargs["top_k"] = top_k
            if seed is not None:
                request_kwargs["seed"] = seed

            request_kwargs["cache_prompt"] = False

            tests = {}
            if test_hello_world:
                tests["hello world"] = lambda server: do_test_hello_world(server, **request_kwargs)
            if test_weather:
                tests["weather"] = lambda server: do_test_weather(server, **request_kwargs)
            if test_calc_result:
                tests["calc result"] = lambda server: do_test_calc_result(server, None, 512, **request_kwargs)

            for test_name, test in tests.items():
                success_count = 0
                failure_count = 0
                failures = []
                success_times = []
                failure_times = []
                logger.info("Running %s (%s, %s): ", test_name, server_name, model)
                for _i in range(n):
                    start_time = time.time()

                    def elapsed():
                        return time.time() - start_time

                    try:
                        test(server)
                        success_times.append(elapsed())
                        success_count += 1
                        logger.info("success")
                    except Exception as e:
                        logger.exception("failure: %s", e)
                        failure_count += 1
                        failure_times.append(elapsed())
                        failures.append(str(e))
                        # import traceback
                        # traceback.print_exc()
                output_file.write(
                    json.dumps(
                        {
                            **output_kwargs,
                            "model": model,
                            "server_name": server_name,
                            "model_id": model_id,
                            "test": test_name,
                            "temp": t,
                            "top_p": top_p,
                            "top_k": top_k,
                            "ctk": ctk,
                            "ctv": ctv,
                            "seed": seed,
                            "success_ratio": float(success_count) / n,
                            "avg_time": mean(success_times + failure_times),
                            "median_time": median(success_times + failure_times),
                            "success_count": success_count,
                            "success_times": success_times,
                            "failure_count": failure_count,
                            "failure_times": failure_times,
                            "failures": list(set(failures)),
                        }
                    )
                    + "\n"
                )
                output_file.flush()

        for t in [None] if temp is None else [t if t >= 0 else None for t in temp]:
            if hf is not None:
                servers: list[tuple[str, str | None]] = [("local-inference-server", None)]
                if llama_baseline is not None:
                    servers.append(("local-inference-server (baseline)", llama_baseline))

                for server_name, server_path in servers:
                    server = ServerProcess()
                    server.n_ctx = n_ctx
                    server.n_slots = 1
                    server.jinja = True
                    server.ctk = ctk
                    server.ctv = ctv
                    server.fa = "on" if fa else "off"
                    server.n_predict = n_predict
                    server.model_hf_repo = hf
                    server.model_hf_file = None
                    server.chat_template = chat_template
                    server.chat_template_file = chat_template_file
                    server.server_path = server_path
                    if port is not None:
                        server.server_port = port
                    # server.debug = True

                    with scoped_server(server):
                        server.start(timeout_seconds=15 * 60)
                        for ignore_chat_grammar in [False]:
                            run(
                                server,
                                server_name=server_name,
                                model_id=hf,
                                temp=t,
                                output_kwargs={
                                    "chat_template": chat_template,
                                    "chat_template_file": chat_template_file,
                                },
                                request_kwargs={
                                    "ignore_chat_grammar": ignore_chat_grammar,
                                },
                            )

            if ollama is not None:
                server = ServerProcess()
                server.server_port = 11434
                server.server_host = "localhost"
                subprocess.check_call(["ollama", "pull", ollama])

                with scoped_server(server):
                    run(
                        server,
                        server_name="ollama",
                        model_id=ollama,
                        temp=t,
                        output_kwargs={
                            "chat_template": None,
                            "chat_template_file": None,
                        },
                        request_kwargs={
                            "model": ollama,
                            "max_tokens": n_predict,
                            "num_ctx": n_ctx,
                        },
                    )


if __name__ == "__main__":
    app()
