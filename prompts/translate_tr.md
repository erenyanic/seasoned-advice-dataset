# Stage 4 prompt — translate to Turkish

One API call per example, run **after** the English reasoning traces exist. All three fields are translated in a single call so the model sees the question, the answer, and the reasoning together and keeps terminology consistent across them.

Substitute `{{GLOSSARY}}` with the contents of `glossary.md`, and the three `{{...}}` content fields from `data/english.jsonl`. Write the result to `data/turkish.jsonl` at the same row index.

---

```
You are translating a cooking and food-science Q&A pair, with its reasoning trace, from English into Turkish.

Use these terminology mappings consistently throughout. Where a term has more than one Turkish rendering, pick by context using the stated rule:

{{GLOSSARY}}

Translate the three fields below.

QUESTION:
{{QUESTION}}

ANSWER:
{{ANSWER}}

REASONING:
{{THINKING}}

Rules:
- Produce natural, fluent Turkish — translate the meaning, not the word order.
- A sentence that is grammatical but reads like translated English is a failure.
- Preserve all markdown structure exactly: headings, bold, italics, lists, code blocks, and link syntax. Translate link text, never link URLs.
- Keep numerals and unit symbols unchanged (180 °C, 350 °F, 2 kg, 15 minutes -> 15 dakika). Do not convert between unit systems.
- Keep proper nouns, brand names, and cited publication names in their original form.
- Translate the reasoning trace in the same voice as the original — it is
  internal reasoning, not prose addressed to a reader.
- Do not add, remove, summarise, or explain anything. The Turkish must carry the same information as the English, no more and no less.

Return exactly this JSON object and nothing else:

{
  "question": "...",
  "answer": "...",
  "thinking": "..."
}
```

---

## Notes on running this

**Volume.** Reasoning traces are often longer than the answers they precede, so translating all three fields is roughly 2–3× the text of translating the Q&A alone. Two things make that manageable:

- **Batch API** — half price for work that isn't latency-sensitive, which this isn't. Submit all 500, poll until `processing_status` is `ended`, then collect results by `custom_id`. Results come back in arbitrary order, so key on `custom_id`, never on position.
- **Prompt caching** — the glossary is identical across all 500 calls. Put it early in the prompt with a cache breakpoint so it is written once and read thereafter. Cache reads cost about a tenth of normal input tokens.

**Structured output.** Rather than parsing the JSON block out of prose, constrain the response shape directly with `output_config.format` and a JSON schema requiring `question`, `answer`, and `thinking`. This removes a whole category of parse failures.

**Validation before you trust it.** Check every row for: all three fields non-empty; markdown structure count (headings, list items, code fences) matching the English; no untranslated English sentences left behind; numerals preserved. Then read about 20 at random yourself — automated checks catch structural damage, not stilted Turkish.
