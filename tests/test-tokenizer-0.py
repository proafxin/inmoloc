import argparse
import pathlib
import time

from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("dir_tokenizer", help="directory containing 'tokenizer.model' file")
parser.add_argument("--fname-tok", help="path to a text file to tokenize", required=True)
args = parser.parse_args()

dir_tokenizer = args.dir_tokenizer
fname_tok = args.fname_tok

tokenizer = AutoTokenizer.from_pretrained(dir_tokenizer)

print("tokenizing file: ", fname_tok)
fname_out = fname_tok + ".tok"
with pathlib.Path(fname_tok).open(encoding="utf-8") as f:
    lines = f.readlines()
    s = "".join(lines)
    t_start = time.time()
    res = tokenizer.encode(s, add_special_tokens=False)  # ty: ignore[unresolved-attribute]
    t_end = time.time()
    print("\nmain : tokenized in", f"{1000.0 * (t_end - t_start):.3f}", "ms (py)")
    with pathlib.Path(fname_out).open("w", encoding="utf-8") as f:
        f.writelines(str(x) + "\n" for x in res)
    print("len(res): ", len(res))
    print("len(lines): ", len(lines))
print("results written to: ", fname_out)
