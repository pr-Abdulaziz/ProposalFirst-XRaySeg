"""Duplicate the pretrained SAM decoder for the Occ bilayer decoder."""

import argparse
import copy
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=Path("pretrained/sam_vit_large/pytorch_model.bin"),
    )
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=Path("pretrained/sam_vit_large/pytorch_model_new.bin"),
    )
    args = parser.parse_args()

    loaded = torch.load(args.input, map_location="cpu")
    state_dict = loaded.get("state_dict", loaded) if isinstance(loaded, dict) else loaded
    if not isinstance(state_dict, dict):
        raise TypeError("SAM checkpoint must contain a state-dict mapping.")

    converted = {}
    decoder_keys = 0
    for key, value in state_dict.items():
        converted[key] = value
        if key.startswith("mask_decoder."):
            converted[key.replace("mask_decoder.", "mask_decoder_bo.", 1)] = (
                copy.deepcopy(value)
            )
            decoder_keys += 1

    if decoder_keys == 0:
        raise KeyError(
            "No mask_decoder.* keys were found; refusing to create an invalid "
            "bilayer checkpoint."
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, args.output)
    print(
        f"Created {args.output}: {len(state_dict)} source keys, "
        f"{decoder_keys} duplicated decoder keys, {len(converted)} total keys."
    )


if __name__ == "__main__":
    main()
