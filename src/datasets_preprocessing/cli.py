import argparse
import json
import sys

from .channels import build_channel
from .config import load_config
from .pipeline import run_pipeline
from .splits import build_splitter


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Split JSONL records and apply a channel transform.")
    parser.add_argument("--config", required=True, help="Path to a TOML configuration file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate configuration and module names only; do not read data or write files")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.dry_run:
            build_splitter(config.split)
            build_channel(config.channel)
            print(f"Configuration valid: {config.input_path} -> {config.output_dir}")
        else:
            counts = run_pipeline(config)
            print(json.dumps({"output_dir": str(config.output_dir), "counts": counts}))
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0
