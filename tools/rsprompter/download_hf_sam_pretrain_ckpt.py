"""Download a Hugging Face SAM checkpoint for offline training."""

import argparse
from pathlib import Path

from transformers import SamModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="facebook/sam-vit-large",
        help="Hugging Face SAM model id (the Occ paper uses ViT-Large).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("pretrained/sam_vit_large"),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download again even when config.json and pytorch_model.bin exist.",
    )
    args = parser.parse_args()

    config_path = args.output_dir / "config.json"
    weight_path = args.output_dir / "pytorch_model.bin"
    if not args.force and config_path.is_file() and weight_path.is_file():
        print(f"SAM assets already exist in {args.output_dir}; skipping download.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = SamModel.from_pretrained(args.model, use_safetensors=False)
    model.save_pretrained(args.output_dir, safe_serialization=False)
    print(f"Saved {args.model} to {args.output_dir}")


if __name__ == "__main__":
    main()
