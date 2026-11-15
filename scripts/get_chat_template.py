#!/usr/bin/env python
"""
Fetches the Jinja chat template of a HuggingFace model.
If a model has multiple chat templates, you can specify the variant name.

Syntax:
  ./scripts/get_chat_template.py model_id [variant]

Examples:
  ./scripts/get_chat_template.py CohereForAI/c4ai-command-r-plus tool_use
  ./scripts/get_chat_template.py microsoft/Phi-3.5-mini-instruct
"""

import json
import pathlib
import re
import sys


def get_chat_template(model_id, variant=None):
    try:
        # Use huggingface_hub library if available.
        # Allows access to gated models if the user has access and ran `huggingface-cli login`.
        from huggingface_hub import hf_hub_download

        config_str = pathlib.Path(hf_hub_download(repo_id=model_id, filename="tokenizer_config.json")).read_text(
            encoding="utf-8"
        )
    except ImportError:
        import requests

        assert re.match(r"^[\w.-]+/[\w.-]+$", model_id), f"Invalid model ID: {model_id}"
        response = requests.get(f"https://huggingface.co/{model_id}/resolve/main/tokenizer_config.json")
        if response.status_code == 401:
            raise Exception(
                "Access to this model is gated, please request access, authenticate with `huggingface-cli login` and make sure to run `pip install huggingface_hub`"
            )
        response.raise_for_status()
        config_str = response.text

    try:
        config = json.loads(config_str)
    except json.JSONDecodeError:
        # Fix https://huggingface.co/NousResearch/Meta-Llama-3-8B-Instruct/blob/main/tokenizer_config.json
        # (Remove extra '}' near the end of the file)
        config = json.loads(re.sub(r'\}([\n\s]*\}[\n\s]*\],[\n\s]*"clean_up_tokenization_spaces")', r"\1", config_str))

    chat_template = config["chat_template"]
    if isinstance(chat_template, str):
        return chat_template
    variants = {ct["name"]: ct["template"] for ct in chat_template}

    def format_variants():
        return ", ".join(f'"{v}"' for v in variants)

    if variant is None:
        if "default" not in variants:
            msg = f"Please specify a chat template variant (one of {format_variants()})"
            raise Exception(msg)
        variant = "default"
        sys.stderr.write(f'Note: picked "default" chat template variant (out of {format_variants()})\n')
    elif variant not in variants:
        msg = f"Variant {variant} not found in chat template (found {format_variants()})"
        raise Exception(msg)

    return variants[variant]


def main(args) -> None:
    if len(args) < 1:
        raise ValueError("Please provide a model ID and an optional variant name")
    model_id = args[0]
    variant = None if len(args) < 2 else args[1]

    template = get_chat_template(model_id, variant)
    sys.stdout.write(template)


if __name__ == "__main__":
    main(sys.argv[1:])
