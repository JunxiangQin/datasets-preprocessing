from dataclasses import replace
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
from random import Random
import tempfile
import unittest

from datasets_preprocessing.channels import IdentityChannel
from datasets_preprocessing.cli import main
from datasets_preprocessing.config import ChannelConfig, PipelineConfig, SplitConfig, load_config
from datasets_preprocessing.pipeline import read_jsonl, run_pipeline
from datasets_preprocessing.splits import RandomSplitter


class SplitTests(unittest.TestCase):
    def test_reproducible_disjoint_and_complete(self):
        records = [{"id": i} for i in range(100)]
        splitter = RandomSplitter(SplitConfig())
        first = splitter.split(records, rng=Random(42))
        self.assertEqual(first, splitter.split(records, rng=Random(42)))
        self.assertNotEqual(first, splitter.split(records, rng=Random(43)))
        self.assertEqual([len(v) for v in first.values()], [80, 10, 10])
        ids = [record["id"] for partition in first.values() for record in partition]
        self.assertEqual(sorted(ids), list(range(100)))
        self.assertEqual(records, [{"id": i} for i in range(100)])

    def test_small_inputs_and_zero_ratio(self):
        for n in range(20):
            split = RandomSplitter(SplitConfig(ratios=(0.7, 0.3, 0))).split(
                [{"id": i} for i in range(n)], rng=Random(7))
            self.assertEqual(len(split["test"]), 0)
            self.assertEqual(sum(map(len, split.values())), n)

    def test_invalid_ratios(self):
        for ratios in ((0.8, 0.1, 0.2), (-0.1, 0.1, 1), (float("nan"), 0, 1),
                       (float("inf"), 0, 0), (1, 0), (True, 0, 0)):
            with self.subTest(ratios=ratios), self.assertRaises(ValueError):
                SplitConfig(ratios=ratios)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "input.jsonl"
        self.records = [{"id": i, "samples": [i, 0]} for i in range(10)]
        self.original = "".join(json.dumps(record) + "\n" for record in self.records)
        self.source.write_text(self.original, encoding="utf-8")
        self.config = PipelineConfig(self.source, self.root / "run")

    def config_file(self, extra=""):
        path = self.root / "config.toml"
        path.write_text('input_path = "input.jsonl"\noutput_dir = "run"\n' + extra,
                        encoding="utf-8")
        return path

    def test_config_paths_and_defaults(self):
        config = load_config(self.config_file())
        self.assertEqual(config, self.config)

    def test_bad_settings(self):
        for extra in ('seed = true', 'seed = "42"', 'typo = 1', '[split]\nname = 2',
                      '[split]\nratios = "bad"', '[split]\nratio = [1, 0, 0]',
                      '[channel]\nparameters = 1', '[channel]\nname = ""'):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                load_config(self.config_file(extra))

    def test_missing_path(self):
        path = self.root / "missing.toml"
        path.write_text('input_path = "input.jsonl"', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "output_dir"):
            load_config(path)

    def test_run_and_manifest_without_changing_source(self):
        self.assertEqual(run_pipeline(self.config), {"train": 8, "validation": 1, "test": 1})
        combined = [record for name in ("train", "validation", "test")
                    for record in read_jsonl(self.config.output_dir / f"{name}.jsonl")]
        self.assertEqual(sorted(combined, key=lambda record: record["id"]), self.records)
        manifest = json.loads((self.config.output_dir / "manifest.json").read_text())
        self.assertEqual(manifest["counts"], {"train": 8, "validation": 1, "test": 1})
        self.assertEqual(manifest["config"]["seed"], 42)
        self.assertEqual(self.source.read_text(), self.original)

    def test_refuses_existing_output(self):
        self.config.output_dir.mkdir()
        sentinel = self.config.output_dir / "keep.txt"
        sentinel.write_text("keep", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            run_pipeline(self.config)
        self.assertEqual(sentinel.read_text(), "keep")
        self.assertEqual(list(self.config.output_dir.iterdir()), [sentinel])

    def test_extension_injection_and_independent_rng(self):
        class NoiseChannel:
            def apply(self, record, *, rng):
                return {**record, "noise": rng.gauss(0, 1)}

        run_pipeline(self.config)
        config = replace(self.config, output_dir=self.root / "noisy",
                         channel=ChannelConfig("test-noise"))
        run_pipeline(config, channel=NoiseChannel())
        again = replace(config, output_dir=self.root / "noisy-again")
        run_pipeline(again, channel=NoiseChannel())
        for name in ("train", "validation", "test"):
            baseline = read_jsonl(self.config.output_dir / f"{name}.jsonl")
            noisy = read_jsonl(config.output_dir / f"{name}.jsonl")
            self.assertEqual([r["id"] for r in baseline], [r["id"] for r in noisy])
            self.assertEqual(noisy, read_jsonl(again.output_dir / f"{name}.jsonl"))

    def test_custom_splitter(self):
        class AllTrain:
            def split(self, records, *, rng):
                return {"train": records, "validation": [], "test": []}

        self.assertEqual(run_pipeline(self.config, splitter=AllTrain()),
                         {"train": 10, "validation": 0, "test": 0})
        self.assertEqual((self.config.output_dir / "test.jsonl").read_text(), "")

    def test_bad_json_and_empty_input_fail_before_output(self):
        for text in ("", "\n  \n", "[]\n", '{"x": NaN}\n', '{"x":\n'):
            with self.subTest(text=text):
                self.source.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    run_pipeline(self.config)
                self.assertFalse(self.config.output_dir.exists())

    def test_identity_does_not_alias_nested_data(self):
        record = {"samples": [1, 2]}
        transformed = IdentityChannel().apply(record, rng=Random(1))
        transformed["samples"][0] = 3
        self.assertEqual(record["samples"], [1, 2])

    def test_cli_dry_run_without_dataset_io(self):
        path = self.config_file()
        self.source.unlink()
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--config", str(path), "--dry-run"]), 0)
        self.assertFalse(self.config.output_dir.exists())

    def test_cli_run(self):
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(["--config", str(self.config_file())]), 0)
        self.assertEqual(json.loads(output.getvalue())["counts"]["train"], 8)

    def test_unknown_module_and_parameters_cli_errors(self):
        for extra in ('[split]\nname = "missing"', '[channel]\nname = "awgn"',
                      '[channel.parameters]\nsnr_db = 10'):
            with self.subTest(extra=extra), redirect_stderr(io.StringIO()) as error:
                self.assertEqual(main(["--config", str(self.config_file(extra)), "--dry-run"]), 1)
                self.assertIn("error:", error.getvalue())
        self.assertFalse(self.config.output_dir.exists())


if __name__ == "__main__":
    unittest.main()
