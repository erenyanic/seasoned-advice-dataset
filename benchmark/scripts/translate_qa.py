#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx>=0.28"]
# ///
"""Translate benchmark question/answer pairs into Turkish.

Benchmark counterpart to scripts/translate.py: same client, same glossary,
same validation approach, but two fields instead of three. Benchmark
references carry no `thinking` field (see benchmark/README.md for why), and
scripts/translate.py hard-requires one on every row, so it cannot run here
unmodified.

Usage:
  uv run benchmark/scripts/translate_qa.py --limit 5    # trial run first
  uv run benchmark/scripts/translate_qa.py
  uv run benchmark/scripts/translate_qa.py --build       # write test_turkish.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
from _deepseek import DeepSeekClient, load_api_key, load_checkpoint, read_jsonl  # noqa: E402

SYSTEM_TEMPLATE = """\
You are translating cooking and food-science content from English into Turkish.

Use these terminology mappings consistently. Where a term has more than one \
Turkish rendering, pick by context using the stated rule:

{glossary}

You will be given a QUESTION and an ANSWER. Translate both.

Rules:
- Produce natural, fluent Turkish. Translate the meaning, not the word order. A \
sentence that is grammatical but reads like translated English is a failure.
- Preserve all markdown structure exactly: headings, bold, italics, lists, code \
blocks, and link syntax. Translate link text, never link URLs.
- Keep numerals and unit symbols unchanged (180 C, 350 F, 2 kg). Do not \
convert between unit systems.
- Keep proper nouns, brand names, and cited publication names in their original \
form.
- Do not add, remove, summarise, or explain anything. The Turkish must carry the \
same information as the English, no more and no less.
- Translate EVERY sentence of EVERY field. Do not condense, abridge, or omit.
- The QUESTION field is a title line followed by the asker's full body text. \
Translate both. Returning only the title is a failure.

Respond with a JSON object containing exactly the keys "question" and "answer", \
whose values are the Turkish translations. Output nothing else.\
"""

# Same structural/length/language checks as scripts/translate.py, generalised
# to an arbitrary field list instead of the fixed (question, answer, thinking)
# triple -- there is no thinking field here.
STRUCTURE_PATTERNS = {
    "headings": re.compile(r"^#{1,6} ", re.M),
    "list items": re.compile(r"^\s*(?:[-*+]|\d+\.) ", re.M),
    "code fences": re.compile(r"```"),
    "links": re.compile(r"\[[^\]]*\]\([^)]*\)"),
}
MIN_LENGTH_RATIO = 0.65
MAX_LENGTH_RATIO = 1.70
ENGLISH_STOPWORDS = frozenset(
    "the and is are of that with for this you your it to in be have on as but "
    "not from can will".split()
)
MAX_ENGLISH_DENSITY = 0.12
MIN_WORDS_FOR_LANGUAGE_CHECK = 40


def english_density(text: str) -> float:
    """Fraction of words that are English function words; ~0 for real Turkish."""
    words = re.findall(r"[a-z']+", text.lower())
    if len(words) < MIN_WORDS_FOR_LANGUAGE_CHECK:
        return 0.0
    return sum(1 for w in words if w in ENGLISH_STOPWORDS) / len(words)


def validation_warnings(question_en: str, answer_en: str, turkish: dict[str, str]) -> list[str]:
    warnings = []
    for field, source in (("question", question_en), ("answer", answer_en)):
        translated = turkish[field]
        for name, pattern in STRUCTURE_PATTERNS.items():
            before, after = len(pattern.findall(source)), len(pattern.findall(translated))
            if name == "list items":
                before = before if before > 1 else 0
                after = after if after > 1 else 0
            if before != after:
                warnings.append(f"{field}: {name} {before} -> {after}")

        ratio = len(translated) / max(1, len(source))
        if not MIN_LENGTH_RATIO <= ratio <= MAX_LENGTH_RATIO:
            warnings.append(f"{field}: length ratio {ratio:.2f} (content dropped?)")

        density = english_density(translated)
        if density > MAX_ENGLISH_DENSITY:
            warnings.append(f"{field}: {density:.0%} English function words (untranslated?)")
    return warnings


def parse_response(raw: str) -> dict[str, str]:
    try:
        parsed = json.loads(raw, strict=False)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.S)
        if not match:
            raise ValueError(f"no JSON object in response: {raw[:200]}")
        parsed = json.loads(match.group(0), strict=False)

    missing = {"question", "answer"} - parsed.keys()
    if missing:
        raise ValueError(f"missing keys in response: {sorted(missing)}")
    for key in ("question", "answer"):
        if not isinstance(parsed[key], str) or not parsed[key].strip():
            raise ValueError(f"field {key!r} is empty or not a string")
    return parsed


def build_user_message(conversation: list[dict]) -> str:
    return f"QUESTION:\n{conversation[0]['content']}\n\nANSWER:\n{conversation[1]['content']}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-file", default="benchmark/data/test_english.jsonl")
    parser.add_argument("--glossary", default="prompts/glossary.md")
    parser.add_argument("--checkpoint", default="benchmark/data/stage_turkish.jsonl")
    parser.add_argument("--out-file", default="benchmark/data/test_turkish.jsonl")
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--reasoning-effort", default="medium",
                        choices=["low", "medium", "high", "max"])
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=8000)
    parser.add_argument("--quality-retries", type=int, default=2,
                        help="Re-request a row whose translation fails validation")
    parser.add_argument("--limit", type=int, help="Process only the first N pending rows")
    parser.add_argument("--build", action="store_true",
                        help="Skip translation; assemble the Turkish test split")
    args = parser.parse_args()

    conversations = read_jsonl(Path(args.in_file))
    ckpt_path = Path(args.checkpoint)
    done = load_checkpoint(ckpt_path)

    if args.build:
        return build_split(conversations, done, Path(args.out_file))

    pending = [i for i in range(len(conversations)) if i not in done]
    if args.limit:
        pending = pending[: args.limit]
    if not pending:
        print("Nothing pending. Run with --build to assemble the Turkish test split.")
        return 0

    system_prompt = SYSTEM_TEMPLATE.format(
        glossary=Path(args.glossary).read_text(encoding="utf-8")
    )
    print(f"{len(done)} already done, {len(pending)} to translate "
          f"(model={args.model}, effort={args.reasoning_effort}, "
          f"concurrency={args.concurrency})")

    client = DeepSeekClient(load_api_key(), args.model, args.reasoning_effort)
    write_lock = threading.Lock()
    counters = {"ok": 0, "warned": 0, "failed": 0}

    def process(index: int) -> None:
        conversation = conversations[index]
        question_en = conversation[0]["content"]
        answer_en = conversation[1]["content"]

        best: dict[str, str] | None = None
        best_warnings: list[str] = []
        attempts = 0

        for attempt in range(args.quality_retries + 1):
            attempts = attempt + 1
            turkish = parse_response(
                client.complete(
                    system_prompt,
                    build_user_message(conversation),
                    max_tokens=args.max_tokens,
                    json_mode=True,
                )
            )
            warnings = validation_warnings(question_en, answer_en, turkish)
            if best is None or len(warnings) < len(best_warnings):
                best, best_warnings = turkish, warnings
            if not warnings:
                break

        turkish, warnings = best, best_warnings
        record = {"index": index, **turkish, "warnings": warnings, "attempts": attempts}

        with write_lock:
            with ckpt_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            counters["warned" if warnings else "ok"] += 1
            total = counters["ok"] + counters["warned"] + counters["failed"]
            print(f"  {total}/{len(pending)}  ok={counters['ok']} "
                  f"warned={counters['warned']} failed={counters['failed']}",
                  end="\r", flush=True)

    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(process, i): i for i in pending}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:  # noqa: BLE001
                    counters["failed"] += 1
                    print(f"\n  row {futures[future]} failed: {exc}", file=sys.stderr)
    finally:
        client.close()

    print(f"\n\nTranslated: {counters['ok']} clean, {counters['warned']} with "
          f"structure warnings, {counters['failed']} failed")
    print(f"Usage: {client.cost_report()}")

    if counters["warned"]:
        print("\nWarnings mean markdown marker counts changed between source and "
              "translation. Review, then delete those lines from "
              f"{ckpt_path} to regenerate.")
    print(f"\nWhen the translations look right:  uv run {sys.argv[0]} --build")
    return 0


def build_split(conversations: list, done: dict, out_path: Path) -> int:
    """Assemble the Turkish test split in the same schema as the English one."""
    missing = [i for i in range(len(conversations)) if i not in done]
    if missing:
        print(f"Refusing to build: {len(missing)} rows are untranslated "
              f"(first few: {missing[:5]}).", file=sys.stderr)
        return 1

    with out_path.open("w", encoding="utf-8") as handle:
        for index in range(len(conversations)):
            turkish = done[index]
            conversation = [
                {"content": turkish["question"], "images": None, "role": "user",
                 "thinking": None, "tool_calls": None},
                {"content": turkish["answer"], "images": None, "role": "assistant",
                 "thinking": None, "tool_calls": None},
            ]
            handle.write(json.dumps(conversation, ensure_ascii=False) + "\n")

    print(f"Wrote {len(conversations)} Turkish conversations to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
