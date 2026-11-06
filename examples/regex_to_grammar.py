import json
import pathlib
import subprocess
import sys

assert len(sys.argv) >= 2
[_, pattern, *rest] = sys.argv

print(
    subprocess.check_output(
        [
            "python",
            str(pathlib.Path(__file__).resolve().parent / "json_schema_to_grammar.py"),
            *rest,
            "-",
            "--raw-pattern",
        ],
        text=True,
        input=json.dumps(
            {
                "type": "string",
                "pattern": pattern,
            },
            indent=2,
        ),
    )
)
