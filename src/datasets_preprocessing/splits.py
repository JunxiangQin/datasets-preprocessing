"""Replace Splitter to add grouped, stratified, or chronological splitting."""

from random import Random
from typing import Any, Protocol

from .config import SplitConfig

Record = dict[str, Any]
SPLIT_NAMES = ("train", "validation", "test")


class Splitter(Protocol):
    def split(self, records: list[Record], *, rng: Random) -> dict[str, list[Record]]:
        """Return train/validation/test partitions without changing the input."""
        ...


class RandomSplitter:
    def __init__(self, config: SplitConfig) -> None:
        self.ratios = config.ratios

    def split(self, records: list[Record], *, rng: Random) -> dict[str, list[Record]]:
        shuffled = list(records)
        rng.shuffle(shuffled)
        exact = [len(shuffled) * ratio for ratio in self.ratios]
        sizes = [int(size) for size in exact]
        # Largest remainder allocation keeps all records and respects zero ratios.
        order = sorted(range(3), key=lambda i: exact[i] - sizes[i], reverse=True)
        for index in order[:len(shuffled) - sum(sizes)]:
            sizes[index] += 1
        train_end = sizes[0]
        validation_end = train_end + sizes[1]
        return dict(zip(SPLIT_NAMES, (
            shuffled[:train_end], shuffled[train_end:validation_end], shuffled[validation_end:]
        )))


def build_splitter(config: SplitConfig) -> Splitter:
    if config.name == "random":
        return RandomSplitter(config)
    raise ValueError(f"Unknown splitter: {config.name!r}; available: random")
