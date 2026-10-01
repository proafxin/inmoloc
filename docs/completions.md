# Completions

Command-line completion is available for some environments.

## Bash Completion

```bash
$ build/bin/local-inference-cli --completion-bash > ~/.local-inference-completion.bash
$ source ~/.local-inference-completion.bash
```

Optionally this can be added to your `.bashrc` or `.bash_profile` to load it
automatically. For example:

```console
$ echo "source ~/.local-inference-completion.bash" >> ~/.bashrc
```
