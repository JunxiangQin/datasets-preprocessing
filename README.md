# datasets-preprocessing

A minimal Python project for reproducible dataset preprocessing. The original
README title above is retained; the sections below describe the new scaffold.

## Requirements and installation

- Python 3.11 or newer (uses the standard-library TOML reader).
- No third-party runtime or test dependencies. Packaging uses `setuptools>=68`,
  declared in `pyproject.toml`; NumPy, SciPy, and audio libraries can be added when
  a concrete processing module needs them.

```bash
git clone https://github.com/JunxiangQin/datasets-preprocessing.git
cd datasets-preprocessing
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e .
```

## Quick start

```bash
# Validate TOML, ratios, and supported module names; no dataset I/O or writes.
datasets-preprocess --config configs/default.toml --dry-run

# Process the ten synthetic example records.
datasets-preprocess --config configs/default.toml

# Equivalent module entry point:
python -m datasets_preprocessing --config configs/default.toml --dry-run
```

The example produces `data/processed/demo/train.jsonl` (8 records),
`validation.jsonl` (1), `test.jsonl` (1), and `manifest.json` containing the
resolved configuration and split counts. Input files are read only. A run fails
if its output directory already exists; choose a new `output_dir` for each run.
The command returns a nonzero status and an error message for invalid input.

## Project layout

| Path | Purpose |
| --- | --- |
| `configs/default.toml` | Configuration entry point and runnable example |
| `src/datasets_preprocessing/config.py` | Typed settings and TOML validation |
| `src/datasets_preprocessing/splits.py` | `Splitter` protocol and random baseline |
| `src/datasets_preprocessing/channels.py` | `Channel` protocol and identity baseline |
| `src/datasets_preprocessing/pipeline.py` | JSONL I/O, split/transform orchestration, manifest |
| `src/datasets_preprocessing/cli.py` | Console command and dry run |
| `data/raw/` | Local source datasets |
| `data/interim/` | Optional intermediate artifacts for future stages |
| `data/processed/` | Generated run directories |
| `examples/sample.jsonl` | Small synthetic fixture safe to commit |
| `tests/` | Standard-library unit and integration tests |
| `.github/workflows/tests.yml` | Installation, test, and CLI checks |

Dataset directories are ignored except for their `.gitkeep` markers. Keep real
datasets out of Git and put small reusable fixtures in `examples/` or `tests/`.

## Configuration and input contract

```toml
input_path = "../data/raw/records.jsonl"
output_dir = "../data/processed/run-001"
seed = 42

[split]
name = "random"
ratios = [0.8, 0.1, 0.1] # train, validation, test; must sum to 1

[channel]
name = "identity"

[channel.parameters]
```

Paths are resolved relative to the TOML file, including when invoked from a
different working directory. Unknown configuration keys and unsupported module
names fail explicitly. Only `random` and `identity` are built in; no AWGN,
Rayleigh/Rician fading, or interference model is implemented yet. Identity
accepts no parameters and preserves record contents.

Input is UTF-8 JSON Lines: one JSON object per nonblank line. Arbitrary fields
are retained; the scaffold does not impose an audio or I/Q representation.
`samples` in the fixture is illustrative. Modules should document their expected
fields, units, sample rate, shape, and real/complex representation. Empty inputs,
malformed JSON, and records that are not objects are rejected.

The baseline shuffles source records with a local seeded RNG, then allocates
counts using the largest remainder method (ties prefer train, then validation,
then test). Small datasets may have empty partitions. Identical input order,
configuration, code, and Python version reproduce a run. Split and channel RNGs
are independent, so changing a channel does not change split membership.

## Extension interfaces

`Splitter.split(records, *, rng)` returns a dictionary with exactly `train`,
`validation`, and `test`, each containing JSON object records. Each input record
must occur in exactly one partition. Implement a grouped, stratified, or temporal
splitter here when independent random records are inappropriate.

`Channel.apply(record, *, rng)` returns a new JSON object and must preserve source
identifiers without modifying the input record. Use the provided RNG rather than
global random state. Channel-specific settings belong in `channel.parameters`.
For CLI support, add the new implementation to `build_channel` (or
`build_splitter`), validate its parameters there, and add a TOML example.

An implementation can first be injected through the Python API:

```python
from copy import deepcopy
from dataclasses import replace
from random import Random

from datasets_preprocessing.config import ChannelConfig, load_config
from datasets_preprocessing.pipeline import run_pipeline
from datasets_preprocessing.splits import Record

class GainChannel:
    def apply(self, record: Record, *, rng: Random) -> Record:
        result = deepcopy(record)
        result["samples"] = [0.5 * x for x in result["samples"]]
        return result

config = load_config("configs/default.toml")
config = replace(
    config,
    output_dir=config.output_dir.with_name("gain-demo"),
    channel=ChannelConfig(name="gain", parameters={"gain": 0.5}),
)
run_pipeline(config, channel=GainChannel())
# A custom splitter can likewise be supplied with splitter=MySplitter(...).
```

Injected implementations own their parameter validation; keep the configuration
consistent with the injected implementation so the manifest describes the run.
The current pipeline applies one channel per record after splitting. Before
adding windowing or multiple noisy variants, split by original recording,
speaker, or source group to keep related samples out of different partitions.
Future fitted transforms should learn their statistics from training data only.

This scaffold loads the dataset into memory. Add streaming/chunked I/O for large
datasets. A successful run writes the manifest last; disk/write failures can
leave an incomplete new run directory, which should be inspected before reuse.

## Development checks

```bash
python -m unittest discover -s tests -v
python -m compileall -q src
```

Tests cover configuration validation, deterministic/disjoint splits, extension
injection, JSONL errors, output protection, and the CLI. CI runs them on Python
3.11 and 3.13 after a normal package installation.
