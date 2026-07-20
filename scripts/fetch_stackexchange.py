#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx>=0.28", "markdownify>=0.13"]
# ///
"""Fetch question/answer pairs from a Stack Exchange site.

Stage 1 of the pipeline: pulls the raw English pairs that later stages
translate into Turkish and augment with reasoning traces.

Selection rules:
  * an answer must be accepted, or otherwise score >= 3
  * the answer's markdown must be between 150 and 3000 characters
  * pairs whose question or answer embeds an image are skipped entirely

Outputs (into --out-dir):
  raw_qa.jsonl   one record per surviving pair, with body text and metadata
  sources.jsonl  per-pair provenance for CC BY-SA attribution

Usage:
  uv run scripts/fetch_stackexchange.py --target 500
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import httpx
from markdownify import markdownify

API = "https://api.stackexchange.com/2.3"

# SE serves bodies as HTML. Any of these means the content leans on a picture
# we are deliberately not scraping, so the pair is unusable in a text dataset.
IMAGE_MARKERS = re.compile(
    r"<img\b|i\.sstatic\.net|i\.stack\.imgur\.com|imgur\.com|<picture\b|<svg\b",
    re.IGNORECASE,
)

MIN_ANSWER_CHARS = 150
MAX_ANSWER_CHARS = 3000
MIN_UNACCEPTED_SCORE = 3


class StackExchangeClient:
    """Thin wrapper that paginates and respects the API's backoff signal."""

    def __init__(self, site: str, pause: float = 0.15) -> None:
        self.site = site
        self.pause = pause
        self.requests_made = 0
        self._client = httpx.Client(timeout=30.0, follow_redirects=True)

    def close(self) -> None:
        self._client.close()

    def get(self, path: str, **params: Any) -> dict[str, Any]:
        params.setdefault("site", self.site)
        response = self._client.get(f"{API}{path}", params=params)
        response.raise_for_status()
        payload = response.json()
        self.requests_made += 1

        # The API asks callers to pause when it returns a backoff, and ignoring
        # it earns a throttle violation on the next call.
        backoff = payload.get("backoff")
        if backoff:
            print(f"  [backoff] sleeping {backoff}s as requested", file=sys.stderr)
            time.sleep(float(backoff) + 0.5)
        else:
            time.sleep(self.pause)

        if payload.get("quota_remaining") is not None and payload["quota_remaining"] < 20:
            print(
                f"  [quota] only {payload['quota_remaining']} requests left today",
                file=sys.stderr,
            )
        return payload

    def paginate(self, path: str, max_items: int, **params: Any) -> Iterator[dict[str, Any]]:
        """Yield items across pages until exhausted or max_items reached."""
        page = 1
        seen = 0
        while seen < max_items:
            payload = self.get(path, page=page, pagesize=100, **params)
            items = payload.get("items", [])
            if not items:
                return
            for item in items:
                yield item
                seen += 1
                if seen >= max_items:
                    return
            if not payload.get("has_more"):
                return
            page += 1


def to_markdown(body: str) -> str:
    """Convert an SE HTML body to markdown, preserving lists and code."""
    # The escape_* flags are off on purpose: markdownify otherwise backslash-
    # escapes punctuation that is plain prose here (producing artefacts like
    # "{\*}"), and those escapes would survive into the translation stage.
    text = markdownify(
        body,
        heading_style="ATX",
        bullets="-",
        escape_asterisks=False,
        escape_underscores=False,
        escape_misc=False,
    )
    # markdownify leaves ragged blank-line runs behind; collapse them.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def has_image(*html_bodies: str) -> bool:
    return any(IMAGE_MARKERS.search(body or "") for body in html_bodies)


def fetch_questions(
    client: StackExchangeClient, sort: str, max_questions: int
) -> list[dict[str, Any]]:
    """Pull candidate questions, newest-first by the chosen sort key."""
    print(f"Fetching up to {max_questions} questions (sort={sort})...")
    questions = list(
        client.paginate(
            "/questions",
            max_items=max_questions,
            order="desc",
            sort=sort,
            filter="withbody",
        )
    )
    print(f"  got {len(questions)} questions")
    return questions


def fetch_answers_by_id(
    client: StackExchangeClient, answer_ids: list[int]
) -> dict[int, dict[str, Any]]:
    """Fetch specific answers in batches of 100, keyed by answer id."""
    answers: dict[int, dict[str, Any]] = {}
    for start in range(0, len(answer_ids), 100):
        batch = answer_ids[start : start + 100]
        ids = ";".join(str(i) for i in batch)
        for answer in client.paginate(
            f"/answers/{ids}", max_items=len(batch) * 2, filter="withbody"
        ):
            answers[answer["answer_id"]] = answer
        print(f"  accepted answers fetched: {len(answers)}/{len(answer_ids)}", end="\r")
    print()
    return answers


def fetch_top_answers(
    client: StackExchangeClient, question_ids: list[int]
) -> dict[int, dict[str, Any]]:
    """For questions with no accepted answer, grab the best-scoring one."""
    best: dict[int, dict[str, Any]] = {}
    for start in range(0, len(question_ids), 100):
        batch = question_ids[start : start + 100]
        ids = ";".join(str(i) for i in batch)
        for answer in client.paginate(
            f"/questions/{ids}/answers",
            max_items=len(batch) * 10,
            order="desc",
            sort="votes",
            filter="withbody",
        ):
            qid = answer["question_id"]
            if qid not in best or answer["score"] > best[qid]["score"]:
                best[qid] = answer
        print(f"  fallback answers scanned: {len(best)}/{len(question_ids)}", end="\r")
    print()
    return best


def build_record(
    index: int, question: dict[str, Any], answer: dict[str, Any], site_host: str
) -> dict[str, Any] | None:
    """Apply the content filters and assemble one pair, or None if rejected."""
    if has_image(question.get("body", ""), answer.get("body", "")):
        return None

    answer_md = to_markdown(answer.get("body", ""))
    if not (MIN_ANSWER_CHARS <= len(answer_md) <= MAX_ANSWER_CHARS):
        return None

    question_md = to_markdown(question.get("body", ""))
    if not question_md:
        return None

    owner = answer.get("owner", {}) or {}
    return {
        "index": index,
        "question_id": question["question_id"],
        "answer_id": answer["answer_id"],
        # The API returns titles HTML-encoded (bodies are decoded by
        # markdownify, but titles never pass through it).
        "title": html.unescape(question["title"]),
        "question_body": question_md,
        "answer_body": answer_md,
        "tags": question.get("tags", []),
        "answer_score": answer["score"],
        "is_accepted": bool(answer.get("is_accepted")),
        "question_url": f"https://{site_host}/q/{question['question_id']}",
        "answer_url": f"https://{site_host}/a/{answer['answer_id']}",
        "answer_author": owner.get("display_name"),
        "answer_author_url": owner.get("link"),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", default="cooking", help="SE site key (default: cooking)")
    parser.add_argument(
        "--site-host",
        default="cooking.stackexchange.com",
        help="Host used to build citation URLs",
    )
    parser.add_argument(
        "--sort",
        default="votes",
        choices=["votes", "activity", "creation", "hot"],
        help="API sort key. 'votes' favours canonical answers; 'activity' "
        "mirrors the site's 'most active' filter (default: votes)",
    )
    parser.add_argument("--target", type=int, default=500, help="Pairs to keep")
    parser.add_argument(
        "--max-questions",
        type=int,
        default=1500,
        help="Candidates to pull before filtering (oversample; default: 1500)",
    )
    parser.add_argument("--out-dir", default="data", help="Directory for outputs")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = StackExchangeClient(args.site)
    try:
        questions = fetch_questions(client, args.sort, args.max_questions)
        by_id = {q["question_id"]: q for q in questions}

        # Accepted answers first: one cheap batched call per 100, and they are
        # the highest-quality targets.
        accepted_ids = [
            q["accepted_answer_id"] for q in questions if q.get("accepted_answer_id")
        ]
        print(f"Fetching {len(accepted_ids)} accepted answers...")
        accepted = fetch_answers_by_id(client, accepted_ids)

        records: list[dict[str, Any]] = []
        for answer in accepted.values():
            question = by_id.get(answer["question_id"])
            if question is None:
                continue
            record = build_record(len(records), question, answer, args.site_host)
            if record:
                records.append(record)
            if len(records) >= args.target:
                break
        print(f"  {len(records)} pairs survived filtering from accepted answers")

        # Only sweep unaccepted questions if we came up short.
        if len(records) < args.target:
            remaining = [
                q["question_id"]
                for q in questions
                if not q.get("accepted_answer_id") and q.get("answer_count", 0) > 0
            ]
            print(f"Short of target; scanning {len(remaining)} questions without an accepted answer...")
            for qid, answer in fetch_top_answers(client, remaining).items():
                if answer["score"] < MIN_UNACCEPTED_SCORE:
                    continue
                question = by_id.get(qid)
                if question is None:
                    continue
                record = build_record(len(records), question, answer, args.site_host)
                if record:
                    records.append(record)
                if len(records) >= args.target:
                    break

        records = records[: args.target]
        for position, record in enumerate(records):
            record["index"] = position

        write_jsonl(out_dir / "raw_qa.jsonl", records)
        write_jsonl(
            out_dir / "sources.jsonl",
            [
                {
                    "index": r["index"],
                    "question_url": r["question_url"],
                    "answer_url": r["answer_url"],
                    "answer_author": r["answer_author"],
                    "answer_author_url": r["answer_author_url"],
                    "license": "CC BY-SA 4.0",
                }
                for r in records
            ],
        )

        accepted_count = sum(1 for r in records if r["is_accepted"])
        print(f"\nWrote {len(records)} pairs to {out_dir}/raw_qa.jsonl")
        print(f"  accepted: {accepted_count}   score>=3 fallback: {len(records) - accepted_count}")
        print(f"  API requests used: {client.requests_made}")
        if len(records) < args.target:
            print(
                f"  WARNING: {len(records)} < target {args.target}; "
                "rerun with a larger --max-questions",
                file=sys.stderr,
            )
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
