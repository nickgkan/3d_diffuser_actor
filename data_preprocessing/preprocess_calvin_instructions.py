"""
Precompute embeddings of instructions.
"""
import os
import re
import json
from pathlib import Path
import itertools
from typing import List, Tuple, Literal, Dict, Optional
import pickle

import tap
import transformers
from tqdm.auto import tqdm
import torch
import numpy as np


TextEncoder = Literal["bert", "clip"]


class Arguments(tap.Tap):
    output: Path
    encoder: TextEncoder = "clip"
    model_max_length: int = 53
    device: str = "cuda"
    verbose: bool = False
    annotation_path: Path


def parse_int(s):
    return int(re.findall(r"\d+", s)[0])


def load_model(encoder: TextEncoder) -> transformers.PreTrainedModel:
    if encoder == "bert":
        model = transformers.BertModel.from_pretrained("bert-base-uncased")
    elif encoder == "clip":
        model = transformers.CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    else:
        raise ValueError(f"Unexpected encoder {encoder}")
    if not isinstance(model, transformers.PreTrainedModel):
        raise ValueError(f"Unexpected encoder {encoder}")
    return model


def load_tokenizer(encoder: TextEncoder) -> transformers.PreTrainedTokenizer:
    if encoder == "bert":
        tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
    elif encoder == "clip":
        tokenizer = transformers.CLIPTokenizer.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
    else:
        raise ValueError(f"Unexpected encoder {encoder}")
    if not isinstance(tokenizer, transformers.PreTrainedTokenizer):
        raise ValueError(f"Unexpected encoder {encoder}")
    return tokenizer


if __name__ == "__main__":
    args = Arguments().parse_args()
    print(args)

    annotations = np.load(str(args.annotation_path), allow_pickle=True).item()
    instructions_string = [s + '.' for s in annotations['language']['ann']]

    tokenizer = load_tokenizer(args.encoder)
    tokenizer.model_max_length = args.model_max_length

    model = load_model(args.encoder)
    model = model.to(args.device)

    instructions = {
        'embeddings': [],
        'text': []
    }

    for instr in tqdm(instructions_string):
        tokens = tokenizer(instr, padding="max_length")["input_ids"]

        tokens = torch.tensor(tokens).to(args.device)
        tokens = tokens.view(1, -1)
        with torch.no_grad():
            pred = model(tokens).last_hidden_state
        instructions['embeddings'].append(pred.cpu())
        instructions['text'].append(instr)

    os.makedirs(str(args.output.parent), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(instructions, f)
