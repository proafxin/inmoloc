import argparse
import pathlib

import torch
from transformers import AutoModel

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", help="Path to GLM model")
args = ap.parse_args()

# find the model part that includes the the multimodal projector weights
model = AutoModel.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
checkpoint = model.state_dict()

# get a list of mm tensor names
mm_tensors = [k for k in checkpoint if k.startswith("vision.adapter.")]

# store these tensors in a new dictionary and torch.save them
projector = {name: checkpoint[name].float() for name in mm_tensors}
torch.save(projector, f"{args.model}/glm.projector")

clip_tensors = [k for k in checkpoint if k.startswith("vision.vit.model.vision_model.")]
if len(clip_tensors) > 0:
    clip = {name.replace("vision.vit.model.", ""): checkpoint[name].float() for name in clip_tensors}
    torch.save(clip, f"{args.model}/glm.clip")

    # added tokens should be removed to be able to convert Mistral models
    if pathlib.Path(f"{args.model}/added_tokens.json").exists():
        pathlib.Path(f"{args.model}/added_tokens.json").write_text("{}\n", encoding="utf-8")

print("Done!")
print(f"Now you can convert {args.model} to a regular LLaMA GGUF file.")
print(f"Also, use {args.model}glm.projector to prepare a glm-encoder.gguf file.")
