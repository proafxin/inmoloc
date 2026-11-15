import argparse
import glob
import pathlib

import torch

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", help="Path to LLaVA v1.5 model")
args = ap.parse_args()

# find the model part that includes the the multimodal projector weights
path = max(glob.glob(f"{args.model}/pytorch_model*.bin"))
checkpoint = torch.load(path)

# get a list of mm tensor names
mm_tensors = [k for k in checkpoint if k.startswith("model.mm_projector")]

# store these tensors in a new dictionary and torch.save them
projector = {name: checkpoint[name].float() for name in mm_tensors}
torch.save(projector, f"{args.model}/llava.projector")

# BakLLaVA models contain CLIP tensors in it
clip_tensors = [k for k in checkpoint if k.startswith("model.vision_tower")]
if len(clip_tensors) > 0:
    clip = {name.replace("vision_tower.vision_tower.", ""): checkpoint[name].float() for name in clip_tensors}
    torch.save(clip, f"{args.model}/llava.clip")

    # added tokens should be removed to be able to convert Mistral models
    if pathlib.Path(f"{args.model}/added_tokens.json").exists():
        pathlib.Path(f"{args.model}/added_tokens.json").write_text("{}\n", encoding="utf-8")

print("Done!")
print(f"Now you can convert {args.model} to a regular LLaMA GGUF file.")
print(f"Also, use {args.model}/llava.projector to prepare a llava-encoder.gguf file.")
