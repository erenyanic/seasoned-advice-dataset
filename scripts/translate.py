#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx>=0.28"]
# ///
"""Stage 4 — translate question, answer, and reasoning into Turkish.

All three fields go in one call per row so the model keeps terminology
consistent across them. The glossary is loaded into the system message, which is
byte-identical on every call and therefore served from DeepSeek's prefix cache
at roughly 1/120th the price of a miss after the first request.

Runs after stage 3: every assistant message must already carry a `thinking`
value, or there is nothing to translate.

Usage:
  uv run scripts/translate.py --limit 20     # trial run first
  uv run scripts/translate.py                # full 500
  uv run scripts/translate.py --build        # write data/turkish.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _deepseek import DeepSeekClient, load_api_key, load_checkpoint, read_jsonl

SYSTEM_TEMPLATE = """\
You are translating cooking and food-science content from English into Turkish.

Use these terminology mappings consistently. Where a term has more than one \
Turkish rendering, pick by context using the stated rule:

{glossary}

You will be given a QUESTION, an ANSWER, and a REASONING trace. Translate all \
three.

Rules:
- Produce natural, fluent Turkish. Translate the meaning, not the word order. A \
sentence that is grammatical but reads like translated English is a failure.
- Preserve all markdown structure exactly: headings, bold, italics, lists, code \
blocks, and link syntax. Translate link text, never link URLs.
- Keep numerals and unit symbols unchanged (180 °C, 350 °F, 2 kg). Do not \
convert between unit systems.
- Keep proper nouns, brand names, and cited publication names in their original \
form.
- Translate the reasoning trace in the same voice as the original: it is \
internal reasoning, not prose addressed to a reader.
- Do not add, remove, summarise, or explain anything. The Turkish must carry the \
same information as the English, no more and no less.
- Translate EVERY sentence of EVERY field. Do not condense, abridge, or omit.
- The QUESTION field is a title line followed by the asker's full body text. \
Translate both. Returning only the title is a failure.
- The REASONING field is often long. Translate it in full, at its original \
length. Compressing it into a shorter summary is a failure.
- Each translated field should come out close to the length of its source. A \
Turkish field markedly shorter than its English original means content was \
dropped.

Respond with a JSON object containing exactly the keys "question", "answer", \
and "thinking", whose values are the Turkish translations. Output nothing else.\
"""

# Structural markers that must survive translation intact. A mismatch means the
# model reformatted the content rather than translating it.
STRUCTURE_PATTERNS = {
    "headings": re.compile(r"^#{1,6} ", re.M),
    "list items": re.compile(r"^\s*(?:[-*+]|\d+\.) ", re.M),
    "code fences": re.compile(r"```"),
    "links": re.compile(r"\[[^\]]*\]\([^)]*\)"),
}


THINKING_ONLY_TEMPLATE = """\
You are translating a cooking and food-science reasoning trace from English into \
Turkish.

Use these terminology mappings consistently:

{glossary}

Translate the entire trace into fluent, natural Turkish. Translate EVERY \
sentence. Do not condense, abridge, summarise, or omit anything. Preserve all \
markdown structure (lists, bold, links) exactly. The output must be comparable \
in length to the input.

Output only the Turkish translation, nothing else.\
"""


def build_user_message(conversation: list[dict]) -> str:
    return (
        f"QUESTION:\n{conversation[0]['content']}\n\n"
        f"ANSWER:\n{conversation[1]['content']}\n\n"
        f"REASONING:\n{conversation[1]['thinking']}"
    )


def parse_response(raw: str) -> dict[str, str]:
    """Parse the JSON object, falling back to brace extraction if wrapped.

    `strict=False` permits literal control characters inside strings. The model
    occasionally emits a raw newline rather than an escaped one mid-string, which
    strict parsing rejects outright — the payload is otherwise well-formed, so
    refusing it would discard a good translation over an escaping detail.
    """
    try:
        parsed = json.loads(raw, strict=False)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.S)
        if not match:
            raise ValueError(f"no JSON object in response: {raw[:200]}")
        parsed = json.loads(match.group(0), strict=False)

    missing = {"question", "answer", "thinking"} - parsed.keys()
    if missing:
        raise ValueError(f"missing keys in response: {sorted(missing)}")
    for key in ("question", "answer", "thinking"):
        if not isinstance(parsed[key], str) or not parsed[key].strip():
            raise ValueError(f"field {key!r} is empty or not a string")
    return parsed


# Turkish renders English at roughly 0.95-1.10x the character count. A field far
# outside that band means content was dropped or invented. Measured on a trial
# run: good fields clustered at 0.95-1.07; dropped-body questions came in at
# 0.05-0.20 and summarised reasoning traces at 0.49-0.52. The band below sits in
# the gap. This catches what marker counts cannot — a field can lose its entire
# body while keeping a marker count of zero on both sides.
MIN_LENGTH_RATIO = 0.65
MAX_LENGTH_RATIO = 1.70

# An untranslated passthrough — the model echoing the English back — is invisible
# to both checks above: the length ratio is ~1.0 and the markdown is identical.
# One row in 500 came back this way and passed everything. English function words
# are the giveaway; genuine Turkish scores near zero on them, the observed
# passthrough scored 0.22-0.27.
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
        return 0.0  # too short to judge; quoted English is normal in short spans
    return sum(1 for w in words if w in ENGLISH_STOPWORDS) / len(words)


def validation_warnings(english: list[dict], turkish: dict[str, str]) -> list[str]:
    """Compare markdown structure and relative length between source and translation."""
    pairs = [
        ("question", english[0]["content"]),
        ("answer", english[1]["content"]),
        ("thinking", english[1]["thinking"]),
    ]
    warnings = []
    for field, source in pairs:
        for name, pattern in STRUCTURE_PATTERNS.items():
            before, after = len(pattern.findall(source)), len(pattern.findall(turkish[field]))
            if name == "list items":
                # Turkish writes ordinals with a trailing period ("18. yüzyılda"
                # = "in the 18th century"), which at line start is byte-identical
                # to a markdown ordered-list item. A real list has more than one
                # entry, so a lone marker is prose, not structure.
                before = before if before > 1 else 0
                after = after if after > 1 else 0
            if before != after:
                warnings.append(f"{field}: {name} {before} -> {after}")

        ratio = len(turkish[field]) / max(1, len(source))
        if not MIN_LENGTH_RATIO <= ratio <= MAX_LENGTH_RATIO:
            warnings.append(f"{field}: length ratio {ratio:.2f} (content dropped?)")

        density = english_density(turkish[field])
        if density > MAX_ENGLISH_DENSITY:
            warnings.append(f"{field}: {density:.0%} English function words (untranslated?)")
    return warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-file", default="data/english.jsonl")
    parser.add_argument("--glossary", default="prompts/glossary.md")
    parser.add_argument("--checkpoint", default="data/stage4_turkish.jsonl")
    parser.add_argument("--out-file", default="data/turkish.jsonl")
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--reasoning-effort", default="medium",
                        choices=["low", "medium", "high", "max"])
    parser.add_argument("--concurrency", type=int, default=8)
    # Reasoning tokens share this budget with the JSON payload, and reasoning
    # length varies per call — at 8000 the payload was truncated mid-object on
    # ~15% of rows, surfacing as "no JSON object in response". The model's
    # ceiling is 384K, and max_tokens is a cap rather than a target, so generous
    # headroom costs nothing.
    parser.add_argument("--max-tokens", type=int, default=32000)
    parser.add_argument("--quality-retries", type=int, default=2,
                        help="Re-request a row whose translation fails validation")
    parser.add_argument("--limit", type=int, help="Process only the first N pending rows")
    parser.add_argument("--build", action="store_true",
                        help="Skip translation; assemble the Turkish split")
    args = parser.parse_args()

    conversations = read_jsonl(Path(args.in_file))
    ckpt_path = Path(args.checkpoint)
    done = load_checkpoint(ckpt_path)

    if args.build:
        return build_split(conversations, done, Path(args.out_file))

    untraced = [i for i, c in enumerate(conversations) if not c[1].get("thinking")]
    if untraced:
        print(f"Refusing to run: {len(untraced)} rows have no reasoning trace "
              f"(first few: {untraced[:5]}). Complete stage 3 and merge it first.",
              file=sys.stderr)
        return 1

    pending = [i for i in range(len(conversations)) if i not in done]
    if args.limit:
        pending = pending[: args.limit]
    if not pending:
        print("Nothing pending. Run with --build to assemble the Turkish split.")
        return 0

    system_prompt = SYSTEM_TEMPLATE.format(
        glossary=Path(args.glossary).read_text(encoding="utf-8")
    )
    thinking_only_prompt = THINKING_ONLY_TEMPLATE.format(
        glossary=Path(args.glossary).read_text(encoding="utf-8")
    )
    print(f"{len(done)} already done, {len(pending)} to translate "
          f"(model={args.model}, effort={args.reasoning_effort}, "
          f"concurrency={args.concurrency})")

    client = DeepSeekClient(load_api_key(), args.model, args.reasoning_effort)
    write_lock = threading.Lock()
    counters = {"ok": 0, "warned": 0, "failed": 0}

    def process(index: int) -> None:
        # Quality failures here are stochastic, not deterministic: long reasoning
        # traces are sometimes returned summarised rather than translated, and the
        # same row translates cleanly on a later attempt. Retrying in place beats
        # making the operator re-run, and keeping the best-scoring attempt means a
        # persistently bad row still yields its least-damaged version.
        best: dict[str, str] | None = None
        best_warnings: list[str] = []
        attempts = 0

        for attempt in range(args.quality_retries + 1):
            attempts = attempt + 1
            turkish = parse_response(
                client.complete(
                    system_prompt,
                    build_user_message(conversations[index]),
                    max_tokens=args.max_tokens,
                    json_mode=True,
                )
            )
            warnings = validation_warnings(conversations[index], turkish)
            if best is None or len(warnings) < len(best_warnings):
                best, best_warnings = turkish, warnings
            if not warnings:
                break

        turkish, warnings = best, best_warnings

        # Long traces are sometimes summarised when all three fields compete for
        # attention in one JSON response. Re-requesting the trace on its own gives
        # it the whole reply — this recovered a 12,000-character trace that came
        # back at 0.18 of source length in the combined call and 1.07 alone.
        # Only fires when `thinking` is the sole remaining problem.
        thinking_only = warnings and all(w.startswith("thinking:") for w in warnings)
        if thinking_only:
            solo = client.complete(
                thinking_only_prompt,
                conversations[index][1]["thinking"],
                max_tokens=args.max_tokens,
            ).strip()
            candidate = {**turkish, "thinking": solo}
            solo_warnings = validation_warnings(conversations[index], candidate)
            if len(solo_warnings) < len(warnings):
                turkish, warnings = candidate, solo_warnings
                attempts += 1

        record = {"index": index, **turkish, "warnings": warnings,
                  "attempts": attempts}

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
              "translation — usually reformatting, occasionally harmless. Review, "
              f"then delete those lines from {ckpt_path} to regenerate.")
    print(f"\nWhen the translations look right:  uv run {sys.argv[0]} --build")
    return 0


def build_split(conversations: list, done: dict, out_path: Path) -> int:
    """Assemble the Turkish split in the same schema as the English one."""
    missing = [i for i in range(len(conversations)) if i not in done]
    if missing:
        print(f"Refusing to build: {len(missing)} rows are untranslated "
              f"(first few: {missing[:5]}).", file=sys.stderr)
        return 1

    with out_path.open("w", encoding="utf-8") as handle:
        for index, english in enumerate(conversations):
            turkish = done[index]
            conversation = [
                {"content": turkish["question"], "images": None, "role": "user",
                 "thinking": None, "tool_calls": None},
                {"content": turkish["answer"], "images": None, "role": "assistant",
                 "thinking": turkish["thinking"], "tool_calls": None},
            ]
            assert len(conversation) == len(english), f"row {index}: message count"
            handle.write(json.dumps(conversation, ensure_ascii=False) + "\n")

    print(f"Wrote {len(conversations)} Turkish conversations to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
