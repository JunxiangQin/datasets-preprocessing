"""Channel extension point for future AWGN, fading, or interference models."""

from copy import deepcopy
from random import Random
from typing import Protocol

from .config import ChannelConfig
from .splits import Record


class Channel(Protocol):
    def apply(self, record: Record, *, rng: Random) -> Record:
        """Return a new record; use rng for randomness and preserve source identifiers."""
        ...


class IdentityChannel:
    def apply(self, record: Record, *, rng: Random) -> Record:
        return deepcopy(record)


def build_channel(config: ChannelConfig) -> Channel:
    if config.name == "identity":
        if config.parameters:
            raise ValueError("The identity channel does not accept parameters")
        return IdentityChannel()
    raise ValueError(f"Unknown channel: {config.name!r}; available: identity")
