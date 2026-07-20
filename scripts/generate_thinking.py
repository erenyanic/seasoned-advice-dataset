#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx>=0.28"]
# ///
"""Stage 3 — generate English reasoning traces with DeepSeek.

One call per example. The trace is written into each assistant message's
`thinking` field; stage 4 then translates question, answer, and trace together.

IMPORTANT — which field becomes the trace:
    The API returns both `content` and `reasoning_content`. We store `content`.
    `reasoning_content` is the model deliberating about *how to write a trace*
    ("the user wants a reasoning trace, I should avoid mentioning the answer")
    — meta-commentary about the task, which is precisely what the guard filter
    below exists to keep out of the dataset. Storing it would poison every row.

Resumable: results append to a checkpoint keyed by row index, so a re-run skips
completed rows. Delete the checkpoint to start over.

Usage:
  uv run scripts/generate_thinking.py --limit 20     # trial run first
  uv run scripts/generate_thinking.py                # full 500
  uv run scripts/generate_thinking.py --merge        # write english.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _deepseek import DeepSeekClient, load_api_key, load_checkpoint, read_jsonl

SYSTEM_PROMPT = """\
You are writing the internal reasoning trace that precedes an answer about \
cooking and food science.

You will be given a QUESTION and its FINAL ANSWER. Write the reasoning process \
that arrives at that answer, in English.

Rules:
- Reason FORWARD toward the answer. Never reference "the answer", "the response \
above", or the fact that an answer already exists. Write as though you are \
working the problem out for the first time.
- Structure it as: understand what is being asked -> recall the relevant cooking \
or food-science knowledge -> weigh the alternatives -> correct yourself where a \
first instinct was wrong -> decide how to structure the response.
- Stay entirely within cooking and food-science reasoning.
- Never mention being an AI, a language model, a developer, a company, training, \
or this task. No meta-commentary of any kind.
- Match the depth of the answer: a two-sentence answer gets a short trace, a \
detailed answer gets a thorough one.
- STOP once you have decided how to structure the response. Never go on to write \
the response itself, or any part of it. The final sentence should be about how \
you intend to present the answer, not the answer's content.
- Never copy or paraphrase sentences from the FINAL ANSWER into the trace. Refer \
to its points in your own reasoning voice.
- Output only the reasoning trace. No preamble, no headings, no closing remarks.\
"""

# Model-identity leakage that must never reach the dataset. Word-boundary
# anchored so "model" does not fire on "modelling the heat transfer".
#
# Deliberately NOT matched: "the user is asking", "the user wants". Those read as
# meta-commentary but are ordinary reasoning-trace register — the reference
# dataset's own traces open with "The user wants to understand what I am".
# Flagging them would false-positive most of the corpus and block the merge.
GUARD = re.compile(
    r"\b(AI|LLM|language model|large language model|as an AI|assistant|"
    r"developer|Anthropic|OpenAI|DeepSeek|Claude|GPT|trained on|training data|"
    r"my training|the prompt)\b",
    re.IGNORECASE,
)

# Traces shorter than this are almost always a refusal or a stub, not reasoning.
MIN_TRACE_CHARS = 120

# A trace that bleeds into answer prose shares a long verbatim run with the
# answer. The threshold comes from the measured distribution over a 20-row trial:
# clean traces shared 16-39 characters (incidental phrases like "the boiling
# point of water"), contaminated ones shared 324 and 1555. 100 sits in that gap.
#
# An earlier version also flagged prose continuing past a "structure the
# response" marker. That signal was dropped: it does not separate the classes.
# Clean traces reached 602 characters of legitimate planning ("I'll include the
# tip about baking upside down...") while a contaminated one sat at 527.
MAX_SHARED_RUN_CHARS = 100


# Slack allowed between the end of a copied block and the end of the trace for
# the copy to still count as "appended". Observed cases had exactly zero.
TERMINAL_SLACK_CHARS = 40


def _longest_shared_run(trace: str, answer: str):
    return SequenceMatcher(None, trace, answer, autojunk=False).find_longest_match(
        0, len(trace), 0, len(answer)
    )


def contamination_reasons(trace: str, answer: str) -> list[str]:
    """Flag a trace that reproduces the answer instead of reasoning toward it."""
    match = _longest_shared_run(trace, answer)
    if match.size > MAX_SHARED_RUN_CHARS:
        return [f"{match.size}-char verbatim run shared with the answer"]
    return []


def repair_contamination(trace: str, answer: str) -> tuple[str, bool]:
    """Strip an answer copy appended to the end of a trace.

    The observed failure is narrow and consistent: the model finishes reasoning,
    ends on a proper "I'll structure the response by..." sentence, then appends
    the answer verbatim with nothing after it. The reasoning above that point is
    sound, so truncating recovers a usable trace instead of paying to re-roll —
    and re-rolling is not reliable, since these rows contaminated again on retry.

    Only the terminal case is repaired. A copy embedded mid-trace means the model
    reasoned *from* the answer text, which truncation cannot fix; those stay
    flagged for regeneration.

    Returns (trace, repaired).
    """
    match = _longest_shared_run(trace, answer)
    if match.size <= MAX_SHARED_RUN_CHARS:
        return trace, False
    if len(trace) - (match.a + match.size) > TERMINAL_SLACK_CHARS:
        return trace, False  # embedded, not appended — not safely repairable

    head = trace[: match.a].rstrip()

    # The copied span may begin mid-sentence, leaving a fragment of the answer's
    # opening glued to the reasoning. Drop trailing sentences found in the answer.
    #
    # The split also accepts a boundary with no whitespace ("method.Butter"),
    # which is exactly how these appends arrive — requiring \s+ let one through.
    for _ in range(5):
        pieces = re.split(r"(?<=[.!?])(?:\s+|(?=[A-Z]))", head)
        if len(pieces) < 2:
            break
        last = pieces[-1].strip()
        if last and len(last) > 15 and last in answer:
            head = " ".join(pieces[:-1]).rstrip()
        else:
            break

    return head, True


def build_user_message(conversation: list[dict]) -> str:
    return (
        f"QUESTION:\n{conversation[0]['content']}\n\n"
        f"FINAL ANSWER:\n{conversation[1]['content']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-file", default="data/english.jsonl")
    parser.add_argument("--checkpoint", default="data/stage3_thinking.jsonl")
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--reasoning-effort", default="medium",
                        choices=["low", "medium", "high", "max"])
    parser.add_argument("--concurrency", type=int, default=8)
    # Shared with reasoning tokens. The longest trace in a full run reached ~3000
    # tokens against a 4000 cap — no truncation occurred, but the margin was thin
    # and a silent cut would return non-empty content rather than an error.
    parser.add_argument("--max-tokens", type=int, default=16000)
    parser.add_argument("--limit", type=int, help="Process only the first N pending rows")
    parser.add_argument("--merge", action="store_true",
                        help="Skip generation; fold the checkpoint into --in-file")
    args = parser.parse_args()

    in_path, ckpt_path = Path(args.in_file), Path(args.checkpoint)
    conversations = read_jsonl(in_path)
    done = load_checkpoint(ckpt_path)

    if args.merge:
        return merge(conversations, done, in_path)

    pending = [i for i in range(len(conversations)) if i not in done]
    if args.limit:
        pending = pending[: args.limit]
    if not pending:
        print("Nothing pending. Run with --merge to write the traces into the dataset.")
        return 0

    print(f"{len(done)} already done, {len(pending)} to generate "
          f"(model={args.model}, effort={args.reasoning_effort}, "
          f"concurrency={args.concurrency})")

    client = DeepSeekClient(load_api_key(), args.model, args.reasoning_effort)
    write_lock = threading.Lock()
    counters = {"ok": 0, "flagged": 0, "failed": 0}

    def process(index: int) -> None:
        trace = client.complete(
            SYSTEM_PROMPT,
            build_user_message(conversations[index]),
            max_tokens=args.max_tokens,
        ).strip()

        answer = conversations[index][1]["content"]
        trace, repaired = repair_contamination(trace, answer)

        hits = sorted({m.group(0).lower() for m in GUARD.finditer(trace)})
        reasons = contamination_reasons(trace, answer)
        if len(trace) < MIN_TRACE_CHARS:
            reasons.append(f"only {len(trace)} chars")
        flagged = bool(hits) or bool(reasons)
        record = {"index": index, "thinking": trace, "flagged": flagged,
                  "guard_hits": hits, "reasons": reasons, "repaired": repaired}

        with write_lock:
            with ckpt_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            counters["flagged" if flagged else "ok"] += 1
            total = counters["ok"] + counters["flagged"] + counters["failed"]
            print(f"  {total}/{len(pending)}  ok={counters['ok']} "
                  f"flagged={counters['flagged']} failed={counters['failed']}",
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

    print(f"\n\nGenerated: {counters['ok']} clean, {counters['flagged']} flagged, "
          f"{counters['failed']} failed")
    print(f"Usage: {client.cost_report()}")

    if counters["flagged"]:
        print(f"\nFlagged rows tripped the guard filter or came back too short.")
        print(f"Inspect them, delete their lines from {ckpt_path}, and re-run to regenerate.")
    if counters["failed"]:
        print(f"{counters['failed']} rows failed; re-run to retry just those.")
    print(f"\nWhen the traces look right:  uv run {sys.argv[0]} --merge")
    return 0


def merge(conversations: list, done: dict, in_path: Path) -> int:
    """Fold checkpointed traces into the assistant `thinking` field."""
    missing = [i for i in range(len(conversations)) if i not in done]
    if missing:
        print(f"Refusing to merge: {len(missing)} rows have no trace yet "
              f"(first few: {missing[:5]}). Run generation to completion first.",
              file=sys.stderr)
        return 1

    flagged = [i for i, r in done.items() if r.get("flagged")]
    if flagged:
        print(f"Refusing to merge: {len(flagged)} rows are still flagged "
              f"(first few: {flagged[:5]}). Regenerate or clear the flag first.",
              file=sys.stderr)
        return 1

    for index, conversation in enumerate(conversations):
        conversation[1]["thinking"] = done[index]["thinking"]

    with in_path.open("w", encoding="utf-8") as handle:
        for conversation in conversations:
            handle.write(json.dumps(conversation, ensure_ascii=False) + "\n")

    print(f"Merged {len(conversations)} traces into {in_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
