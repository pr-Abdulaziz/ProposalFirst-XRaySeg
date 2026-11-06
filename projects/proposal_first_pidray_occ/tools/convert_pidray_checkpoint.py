"""Convert a released MMDetection 2.x PIDray checkpoint for this project."""

import argparse
from pathlib import Path

import torch

R0_PREFIXES = (
    "confidence_transformer.",
    "confidence_fc.",
    "query_embed.",
    "positional_encoding.",
)


def convert(state_dict: dict, target: str = "hybrid") -> dict:
    if not isinstance(state_dict, dict):
        raise TypeError("Expected a state-dict mapping in the input checkpoint.")
    converted = {}
    for key, value in state_dict.items():
        key = key.removeprefix("module.")
        if target == "hybrid" and key.startswith("roi_head.mask_head."):
            continue
        new_key = f"r0_head.{key}" if key.startswith(R0_PREFIXES) else key
        if new_key in converted:
            raise KeyError(f"Checkpoint-key collision: {new_key}")
        converted[new_key] = value
    return converted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--target",
        choices=("baseline", "hybrid"),
        default="hybrid",
        help="Hybrid conversion drops the incompatible legacy FCN mask head.",
    )
    parser.add_argument(
        "--training-scope",
        choices=("official_full_train",),
        default="official_full_train",
        help=(
            "Data scope used to train the input checkpoint. Released PIDray "
            "weights use the default official_full_train value."
        ),
    )
    args = parser.parse_args()

    loaded = torch.load(args.input, map_location="cpu")
    if isinstance(loaded, dict) and "state_dict" in loaded:
        checkpoint = loaded
        state_dict = checkpoint["state_dict"]
        checkpoint["state_dict"] = convert(state_dict, args.target)
    else:
        state_dict = loaded
        checkpoint = {"state_dict": convert(state_dict, args.target)}
    checkpoint.setdefault("meta", {})["pidray_conversion_target"] = args.target
    checkpoint["meta"]["pidray_training_scope"] = args.training_scope
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    converted_count = len(checkpoint["state_dict"])
    print(
        f"Converted {converted_count}/{len(state_dict)} parameters "
        f"for {args.target} to {args.output}"
    )


if __name__ == "__main__":
    main()
