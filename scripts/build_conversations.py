#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Shape scraped pairs into the target conversation schema.

Stage 2 of the pipeline: turns `raw_qa.jsonl` (bodies plus metadata) into the
two-message format the dataset ships in. Runs before translation so that later
stages only ever rewrite field *values*, never the structure.

The user message is the question title joined to its body; the assistant message
is the answer. `thinking` stays null here and is filled by the reasoning pass.

Usage:
  uv run scripts/build_conversations.py --split english
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# Fixed key order matching the reference schema. Every record carries all five
# keys, and nothing else, so the file loads with standard chat templates.
MESSAGE_KEYS = ("content", "images", "role", "thinking", "tool_calls")


def message(role: str, content: str, thinking: str | None = None) -> dict[str, Any]:
    return {
        "content": content,
        "images": None,
        "role": role,
        "thinking": thinking,
        "tool_calls": None,
    }


def build_user_content(record: dict[str, Any]) -> str:
    """Join title and body, skipping the body when it merely repeats the title."""
    title = record["title"].strip()
    body = record["question_body"].strip()
    if not body or body.lower() == title.lower():
        return title
    return f"{title}\n\n{body}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-file", default="data/raw_qa.jsonl")
    parser.add_argument("--out-file", default=None, help="Defaults to data/<split>.jsonl")
    parser.add_argument("--split", default="english", help="Split name (default: english)")
    args = parser.parse_args()

    in_path = Path(args.in_file)
    out_path = Path(args.out_file) if args.out_file else Path("data") / f"{args.split}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = [json.loads(line) for line in in_path.read_text(encoding="utf-8").splitlines() if line]

    conversations = [
        [
            message("user", build_user_content(record)),
            message("assistant", record["answer_body"].strip()),
        ]
        for record in records
    ]

    # Fail loudly rather than shipping a subtly wrong schema downstream.
    for position, conversation in enumerate(conversations):
        assert len(conversation) == 2, f"row {position}: expected 2 messages"
        assert [m["role"] for m in conversation] == ["user", "assistant"], (
            f"row {position}: wrong role order"
        )
        for msg in conversation:
            assert tuple(msg) == MESSAGE_KEYS, f"row {position}: key mismatch {tuple(msg)}"
            assert msg["content"].strip(), f"row {position}: empty content"

    with out_path.open("w", encoding="utf-8") as handle:
        for conversation in conversations:
            handle.write(json.dumps(conversation, ensure_ascii=False) + "\n")

    print(f"Wrote {len(conversations)} conversations to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
