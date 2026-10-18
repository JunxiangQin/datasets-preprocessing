"""Small in-memory JSONL pipeline; replace I/O for large datasets."""

from dataclasses import asdict
import json
from pathlib import Path
from random import Random

from .channels import Channel, build_channel
from .config import PipelineConfig
from .splits import Record, SPLIT_NAMES, Splitter, build_splitter


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON number: {value}")


def read_jsonl(path: Path) -> list[Record]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line, parse_constant=_reject_constant)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: each record must be a JSON object")
            records.append(record)
    if not records:
        raise ValueError(f"Input contains no records: {path}")
    return records


def run_pipeline(
    config: PipelineConfig, *, splitter: Splitter | None = None, channel: Channel | None = None
) -> dict[str, int]:
    splitter = splitter if splitter is not None else build_splitter(config.split)
    channel = channel if channel is not None else build_channel(config.channel)
    # Split source records before generating channel variants to avoid leakage.
    partitions = splitter.split(read_jsonl(config.input_path), rng=Random(config.seed))
    if set(partitions) != set(SPLIT_NAMES):
        raise ValueError("Splitter must return exactly train, validation, and test")
    # Independent RNGs keep channel changes from changing split membership.
    encoded = {}
    counts = {}
    for name in SPLIT_NAMES:
        rng = Random(f"{config.seed}:channel:{name}")
        transformed = [channel.apply(record, rng=rng) for record in partitions[name]]
        if any(not isinstance(record, dict) for record in transformed):
            raise ValueError("Channel must return JSON object records")
        encoded[name] = "".join(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                                for record in transformed)
        counts[name] = len(transformed)
    manifest = {"config": asdict(config), "counts": counts}
    manifest["config"]["input_path"] = str(config.input_path)
    manifest["config"]["output_dir"] = str(config.output_dir)
    manifest_text = json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    # Only create a new run directory; an existing directory is never overwritten.
    config.output_dir.mkdir(parents=True, exist_ok=False)
    for name in SPLIT_NAMES:
        (config.output_dir / f"{name}.jsonl").write_text(encoded[name], encoding="utf-8")
    (config.output_dir / "manifest.json").write_text(manifest_text, encoding="utf-8")
    return counts
