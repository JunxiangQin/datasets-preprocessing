"""Load validated TOML configuration using only the Python standard library."""

from dataclasses import dataclass, field
import math
from pathlib import Path
import tomllib
from typing import Any


@dataclass(frozen=True)
class SplitConfig:
    name: str = "random"
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("split.name must be a nonempty string")
        if (
            len(self.ratios) != 3
            or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
                   for v in self.ratios)
            or not math.isclose(sum(self.ratios), 1.0, rel_tol=0, abs_tol=1e-9)
        ):
            raise ValueError("split.ratios must contain three finite, nonnegative values summing to 1")


@dataclass(frozen=True)
class ChannelConfig:
    name: str = "identity"
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("channel.name must be a nonempty string")
        if not isinstance(self.parameters, dict):
            raise ValueError("channel.parameters must be a table")


@dataclass(frozen=True)
class PipelineConfig:
    input_path: Path
    output_dir: Path
    seed: int = 42
    split: SplitConfig = field(default_factory=SplitConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)

    def __post_init__(self) -> None:
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")


def _table(value: Any, allowed: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a table")
    unknown = value.keys() - allowed
    if unknown:
        raise ValueError(f"Unknown {label} settings: {', '.join(sorted(unknown))}")
    return value


def load_config(path: str | Path) -> PipelineConfig:
    path = Path(path).resolve()
    with path.open("rb") as handle:
        raw = _table(tomllib.load(handle),
                     {"input_path", "output_dir", "seed", "split", "channel"}, "config")
    paths = {}
    for key in ("input_path", "output_dir"):
        value = raw.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a nonempty path string")
        paths[key] = (path.parent / value).resolve()
    split = _table(raw.get("split", {}), {"name", "ratios"}, "split")
    ratios = split.get("ratios", [0.8, 0.1, 0.1])
    if not isinstance(ratios, list):
        raise ValueError("split.ratios must be an array")
    channel = _table(raw.get("channel", {}), {"name", "parameters"}, "channel")
    return PipelineConfig(
        **paths, seed=raw.get("seed", 42),
        split=SplitConfig(name=split.get("name", "random"), ratios=tuple(ratios)),
        channel=ChannelConfig(**channel),
    )
