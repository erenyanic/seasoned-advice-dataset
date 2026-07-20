#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx>=0.28"]
# ///
"""Repair reasoning traces that stage 4 returned summarised instead of translated.

A minority of long traces come back at roughly half their source length: the
model condenses rather than translates when handed a large block, and it does so
persistently — the same row compresses again across repeated attempts and even
when the trace is re-requested on its own.

Chunking removes the cause rather than re-rolling against it. The trace is split
on paragraph boundaries into pieces small enough that summarising is not
tempting, each is translated independently, and the pieces are rejoined with the
original separators. Same model and glossary as stage 4, so the repaired rows
keep the voice of the other 495.

Usage:
  uv run scripts/repair_long_traces.py            # auto-detect failing rows
  uv run scripts/repair_long_traces.py --rows 9 43
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _deepseek import DeepSeekClient, load_api_key, load_checkpoint, read_jsonl
from translate import validation_warnings

CHUNK_PROMPT = """\
You are translating an excerpt from a cooking and food-science reasoning trace \
from English into Turkish.

Use these terminology mappings consistently:

{glossary}

This is one excerpt from a longer trace. Translate it completely and literally \
into fluent, natural Turkish. Translate every sentence. Do not condense, \
summarise, omit, or add anything. Do not add an introduction or conclusion — \
the excerpt continues from and into other text. Preserve markdown structure \
exactly.

Output only the Turkish translation of the excerpt.\
"""

MAX_CHUNK_CHARS = 1200


def split_into_chunks(text: str, limit: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split on blank lines, packing paragraphs up to the limit.

    Paragraph boundaries are preferred because they rarely fall mid-argument, so
    each chunk stays independently translatable. An oversized single paragraph is
    emitted alone rather than cut mid-sentence.
    """
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if current and len(current) + len(paragraph) + 2 > limit:
            chunks.append(current)
            current = paragraph
        else:
            current = f"{current}\n\n{paragraph}" if current else paragraph
    if current:
        chunks.append(current)
    return chunks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-file", default="data/english.jsonl")
    parser.add_argument("--checkpoint", default="data/stage4_turkish.jsonl")
    parser.add_argument("--glossary", default="prompts/glossary.md")
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--rows", type=int, nargs="*", help="Row indices to repair")
    args = parser.parse_args()

    conversations = read_jsonl(Path(args.in_file))
    ckpt_path = Path(args.checkpoint)
    done = load_checkpoint(ckpt_path)

    if args.rows:
        targets = args.rows
    else:
        targets = [
            i for i, r in done.items()
            if any(w.startswith("thinking:") for w in r.get("warnings", []))
        ]
    if not targets:
        print("No rows need repair.")
        return 0

    prompt = CHUNK_PROMPT.format(glossary=Path(args.glossary).read_text(encoding="utf-8"))
    client = DeepSeekClient(load_api_key(), args.model, args.reasoning_effort)
    repaired = 0

    try:
        for index in sorted(targets):
            source = conversations[index][1]["thinking"]
            chunks = split_into_chunks(source)
            pieces = [
                client.complete(prompt, chunk, max_tokens=8000).strip() for chunk in chunks
            ]
            joined = "\n\n".join(pieces)

            candidate = {**done[index], "thinking": joined}
            warnings = validation_warnings(conversations[index], candidate)
            ratio = len(joined) / max(1, len(source))
            status = "OK" if not warnings else f"still warned: {warnings}"
            print(f"idx {index}: {len(chunks)} chunks | "
                  f"{len(done[index]['thinking'])} -> {len(joined)} chars "
                  f"(ratio {ratio:.2f}) | {status}")

            if len(warnings) < len(done[index].get("warnings", [])) or not warnings:
                done[index] = {**candidate, "warnings": warnings, "repaired_by": "chunked"}
                repaired += 1
    finally:
        client.close()

    with ckpt_path.open("w", encoding="utf-8") as handle:
        for index in sorted(done):
            handle.write(json.dumps(done[index], ensure_ascii=False) + "\n")

    print(f"\nRepaired {repaired}/{len(targets)} rows")
    print(f"Usage: {client.cost_report()}")
    print("Rebuild the split with:  uv run scripts/translate.py --build")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
