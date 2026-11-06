## gguf

This is a Python package for writing binary files in the [GGUF](https://github.com/ggml-org/ggml/pull/302)
(GGML Universal File) format.

See [scripts/convert_hf_to_gguf.py](../scripts/convert_hf_to_gguf.py) as an example for its usage.

## Installation

The package is part of the `inmoloc` project at the root of the repository. From the root:

```sh
uv sync
```

The `gui` extra adds the visual GGUF editor:

```sh
uv sync --extra gui
```

## API Examples/Simple Tools

[examples/writer.py](examples/writer.py) — Generates `example.gguf` in the current directory to demonstrate generating a GGUF file. Note that this file cannot be used as a model.

[examples/reader.py](examples/reader.py) — Extracts and displays key-value pairs and tensor details from a GGUF file in a readable format.

[scripts/gguf_dump.py](scripts/gguf_dump.py) (`uv run gguf-dump`) — Dumps a GGUF file's metadata to the console.

[scripts/gguf_set_metadata.py](scripts/gguf_set_metadata.py) (`uv run gguf-set-metadata`) — Allows changing simple metadata values in a GGUF file by key.

[scripts/gguf_convert_endian.py](scripts/gguf_convert_endian.py) (`uv run gguf-convert-endian`) — Allows converting the endianness of GGUF files.

[scripts/gguf_new_metadata.py](scripts/gguf_new_metadata.py) (`uv run gguf-new-metadata`) — Copies a GGUF file with added/modified/removed metadata values.

[scripts/gguf_editor_gui.py](scripts/gguf_editor_gui.py) (`uv run gguf-editor-gui`) — Allows for viewing, editing, adding, or removing metadata values within a GGUF file as well as viewing its tensors with a Qt interface.

## Development

`uv sync` installs the project in editable mode: changes to `gguf/` apply without reinstalling.

## Building

```sh
uv build
```

## Run Unit Tests

From the root of the repository:

```bash
uv run python -m unittest discover ./tests/gguf -v
```
