"""Shared DeepSeek client helpers for the generation stages.

Not a stage itself — imported by `generate_thinking.py` and `translate.py`.

Two things here are load-bearing forz cost and correctness:

* **Prefix caching.** DeepSeek caches identical prompt prefixes automatically and
  bills a cache hit at roughly 1/120th of a miss. Every caller therefore puts its
  stable instruction block in the system message and only the per-row content in
  the user message, so the prefix is byte-identical across all 500 calls.

* **`content` vs `reasoning_content`.** The response carries both. `content` is
  what the model deliberately wrote; `reasoning_content` is its own internal
  deliberation about how to write it. Only `content` is ever stored — see the
  note in `generate_thinking.py` for why conflating them would poison the data.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import httpx

API_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_MODEL = "deepseek-v4-pro"
RETRY_STATUS = {408, 429, 500, 502, 503, 504}


def load_api_key(env_path: Path | None = None) -> str:
    """Read the DeepSeek key from the environment or a .env file.

    Parsed with a regex rather than split("=") because the key line may carry
    whitespace around the separator, which `source .env` would reject outright.
    """
    for name in ("DEEPSEEK_API", "DEEPSEEK_API_KEY"):
        if os.environ.get(name):
            return os.environ[name].strip()

    env_path = env_path or Path(".env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            match = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)", line)
            if match and "DEEPSEEK" in match.group(1).upper():
                return match.group(2).strip().strip("\"'")

    raise SystemExit(
        "No DeepSeek API key found. Set DEEPSEEK_API in the environment or .env"
    )


class DeepSeekClient:
    """Thin chat-completions wrapper with retry and usage accounting."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        reasoning_effort: str = "medium",
        max_retries: int = 5,
        timeout: float = 300.0,
    ) -> None:
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_retries = max_retries
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._client = httpx.Client(timeout=timeout)
        # Accumulated across the run so the caller can report real spend.
        self.usage = {
            "calls": 0,
            "cache_hit_tokens": 0,
            "cache_miss_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
        }

    def close(self) -> None:
        self._client.close()

    def complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 8000,
        json_mode: bool = False,
    ) -> str:
        """Return the assistant's `content`, retrying transient failures."""
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                # System first and byte-identical per stage — this is the
                # cacheable prefix. Never interpolate per-row data into it.
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "thinking": {"type": "enabled"},
            "reasoning_effort": self.reasoning_effort,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        last_error = ""
        for attempt in range(self.max_retries):
            try:
                response = self._client.post(API_URL, headers=self._headers, json=body)
            except httpx.RequestError as exc:
                last_error = f"network error: {exc}"
            else:
                if response.status_code == 200:
                    payload = response.json()
                    self._record_usage(payload.get("usage", {}))
                    content = payload["choices"][0]["message"].get("content")
                    if content:
                        return content
                    last_error = "empty content in response"
                elif response.status_code in RETRY_STATUS:
                    last_error = f"HTTP {response.status_code}"
                else:
                    # 400/401/403 will not fix themselves — fail loudly.
                    raise RuntimeError(
                        f"DeepSeek returned {response.status_code}: {response.text[:400]}"
                    )

            # Full jitter backoff; a thundering herd of retries is what turns a
            # brief 429 into a run-ending one.
            delay = min(60.0, 2.0**attempt) * (0.5 + random.random() / 2)
            time.sleep(delay)

        raise RuntimeError(f"giving up after {self.max_retries} attempts: {last_error}")

    def _record_usage(self, usage: dict[str, Any]) -> None:
        self.usage["calls"] += 1
        self.usage["cache_hit_tokens"] += usage.get("prompt_cache_hit_tokens", 0)
        self.usage["cache_miss_tokens"] += usage.get("prompt_cache_miss_tokens", 0)
        self.usage["output_tokens"] += usage.get("completion_tokens", 0)
        details = usage.get("completion_tokens_details") or {}
        self.usage["reasoning_tokens"] += details.get("reasoning_tokens", 0)

    def cost_report(self) -> str:
        """Estimate spend from published deepseek-v4-pro rates (USD / 1M tokens)."""
        hit, miss = self.usage["cache_hit_tokens"], self.usage["cache_miss_tokens"]
        out = self.usage["output_tokens"]
        total = (hit * 0.003625 + miss * 0.435 + out * 0.87) / 1e6
        return (
            f"{self.usage['calls']} calls | "
            f"input {miss:,} miss + {hit:,} cached | "
            f"output {out:,} (incl. {self.usage['reasoning_tokens']:,} reasoning) | "
            f"~${total:.2f}"
        )


def read_jsonl(path: Path) -> list[Any]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_checkpoint(path: Path) -> dict[int, dict[str, Any]]:
    """Read completed rows so an interrupted run resumes instead of restarting."""
    if not path.exists():
        return {}
    done: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue  # tolerate a half-written final line from a hard kill
        done[record["index"]] = record
    return done
