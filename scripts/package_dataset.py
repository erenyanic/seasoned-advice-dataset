#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["datasets>=4.0"]
# ///
"""Stage 5 — package the JSONL splits into parquet.

Produces the release layout: one parquet file per split, each holding a single
column named `train` whose value is the two-message conversation list. Each
parquet is written beside its source JSONL, so the training splits land in
`data/` and the held-out benchmark splits in `benchmark/data/`, matching the
paths declared in the README's `configs:` front matter.

The feature types are declared explicitly rather than inferred. `images` and
`tool_calls` are `null` in every row, and while arrow would infer the `null` type
from the data anyway, stating it makes the contract explicit and fails loudly if
a future run ever puts a value there.

All four splits share one schema, which is what lets them sit in a single Hub
config. `thinking` stays `Value("string")` even though it is null in every
benchmark row — an arrow string column is nullable, so an all-null column is
valid, and declaring it `null` instead would make the benchmark schema
incompatible with the training one.

Usage:
  uv run scripts/package_dataset.py
  uv run scripts/package_dataset.py --verify   # inspect what was written
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, List, Value

MESSAGE_FEATURES = {
    "content": Value("string"),
    "images": Value("null"),
    "role": Value("string"),
    "thinking": Value("string"),
    "tool_calls": Value("null"),
}
FEATURES = Features({"train": List(MESSAGE_FEATURES)})

# Source JSONL per split. The parquet is written next to its source, so the
# benchmark splits stay under benchmark/ rather than being flattened into data/.
SPLITS = {
    "english": "data/english.jsonl",
    "turkish": "data/turkish.jsonl",
    "test_english": "benchmark/data/test_english.jsonl",
    "test_turkish": "benchmark/data/test_turkish.jsonl",
}


def parquet_path(split_name: str) -> Path:
    return Path(SPLITS[split_name]).parent / f"{split_name}-00000-of-00001.parquet"


def load_split(path: Path) -> Dataset:
    rows = [
        {"train": json.loads(line)}
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return Dataset.from_list(rows, features=FEATURES)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true",
                        help="Load the written parquet back and print its schema")
    args = parser.parse_args()

    if args.verify:
        loaded = DatasetDict({
            name: Dataset.from_parquet(str(parquet_path(name))) for name in SPLITS
        })
        print(loaded)
        print(loaded["english"].features)
        first = loaded["turkish"][0]["train"]
        print(f"\nturkish[0] roles: {[m['role'] for m in first]}")
        print(f"turkish[0] user content: {first[0]['content'][:80]}...")
        print(f"turkish[0] thinking set: {first[1]['thinking'] is not None}")

        # The held-out splits carry no reasoning trace by design; assert it here
        # rather than trusting it, since a stray trace would mean the benchmark
        # was built from the training pipeline by mistake.
        for name in ("test_english", "test_turkish"):
            traced = [
                i for i, row in enumerate(loaded[name])
                if any(m["thinking"] is not None for m in row["train"])
            ]
            status = "none (correct)" if not traced else f"UNEXPECTED at rows {traced[:5]}"
            print(f"{name}: {len(loaded[name])} rows, reasoning traces: {status}")

        schemas = {name: loaded[name].features for name in SPLITS}
        identical = len({str(f) for f in schemas.values()}) == 1
        print(f"\nAll four splits share one schema: {identical}")
        return 0

    dataset = DatasetDict({name: load_split(Path(path)) for name, path in SPLITS.items()})
    print(dataset)

    for name, split in dataset.items():
        target = parquet_path(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        split.to_parquet(str(target))
        size_mb = target.stat().st_size / 1e6
        print(f"wrote {target}  ({len(split)} rows, {size_mb:.2f} MB)")

    print("\nVerify with:  uv run scripts/package_dataset.py --verify")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
