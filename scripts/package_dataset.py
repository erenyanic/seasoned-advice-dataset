#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["datasets>=4.0"]
# ///
"""Stage 5 — package the JSONL splits into parquet.

Produces the release layout: one parquet file per split, each holding a single
column named `train` whose value is the two-message conversation list.

The feature types are declared explicitly rather than inferred. `images` and
`tool_calls` are `null` in every row, and while arrow would infer the `null` type
from the data anyway, stating it makes the contract explicit and fails loudly if
a future run ever puts a value there.

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

SPLITS = {"english": "data/english.jsonl", "turkish": "data/turkish.jsonl"}


def load_split(path: Path) -> Dataset:
    rows = [
        {"train": json.loads(line)}
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return Dataset.from_list(rows, features=FEATURES)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--verify", action="store_true",
                        help="Load the written parquet back and print its schema")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    if args.verify:
        loaded = DatasetDict({
            name: Dataset.from_parquet(str(out_dir / f"{name}-00000-of-00001.parquet"))
            for name in SPLITS
        })
        print(loaded)
        print(loaded["english"].features)
        first = loaded["turkish"][0]["train"]
        print(f"\nturkish[0] roles: {[m['role'] for m in first]}")
        print(f"turkish[0] user content: {first[0]['content'][:80]}...")
        print(f"turkish[0] thinking set: {first[1]['thinking'] is not None}")
        return 0

    dataset = DatasetDict({name: load_split(Path(path)) for name, path in SPLITS.items()})
    print(dataset)

    for name, split in dataset.items():
        target = out_dir / f"{name}-00000-of-00001.parquet"
        split.to_parquet(str(target))
        size_mb = target.stat().st_size / 1e6
        print(f"wrote {target}  ({len(split)} rows, {size_mb:.2f} MB)")

    print("\nVerify with:  uv run scripts/package_dataset.py --verify")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
