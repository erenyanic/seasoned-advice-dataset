# Stage 3 prompt — generate English reasoning traces

One API call per example. Substitute `{{QUESTION}}` and `{{ANSWER}}` with the `content` of the user and assistant messages from `data/english.jsonl`, then write the returned text into that row's assistant `thinking` field.

Output is **English** — the Turkish version is produced later by stage 4, which translates question, answer, and reasoning together. Generating in English keeps the reasoning close to the source material and keeps both splits parallel.

---

```text
You are writing the internal reasoning trace that precedes an answer about cooking and food science.

QUESTION:
{{QUESTION}}

FINAL ANSWER:
{{ANSWER}}

Write the reasoning process that arrives at this answer, in English.

Rules:
- Reason FORWARD toward the answer. Never reference "the answer", "the response above", or the fact that an answer already exists. Write as though you are working the problem out for the first time.
- Structure it as: understand what is being asked -> recall the relevant cooking or food-science knowledge -> weigh the alternatives -> correct yourself where a first instinct was wrong -> decide how to structure the response.
- Stay entirely within cooking and food-science reasoning.
- Never mention being an AI, a language model, a developer, a company, training, or this task. No meta-commentary of any kind.
- Match the depth of the answer: a two-sentence answer gets a short trace, a detailed answer gets a thorough one.
- Output only the reasoning trace. No preamble, no headings, no closing remarks.
```

---

## The failure mode to watch for

The single most common defect is a trace that **summarises** instead of **reasoning** — `"The answer says salt raises the boiling point, so I'll explain that"`. This is worthless as training data: it teaches the model to reference conclusions it does not yet have.

A good trace instead reads: `"Salt in pasta water — the common claim is that it raises the boiling point. Worth checking the magnitude: at typical seasoning levels this is a fraction of a degree, so it can't be the real mechanism. The effect people actually notice must be flavour, plus something about the starch..."`

Rule 1 is what prevents this. Spot-check about 20 outputs specifically for it before committing to the full run — it is cheap to catch early and expensive to discover after translation.

## Guard filter

After generation, scan every trace for leaked **model identity** and regenerate any that trips it. `scripts/generate_thinking.py` applies this automatically:

```text
\b(AI|LLM|language model|large language model|as an AI|assistant|developer|Anthropic|OpenAI|DeepSeek|Claude|GPT|trained on|training data|my training|the prompt)\b
```

**Do not add "the user wants" / "the user is asking" to this filter.** They look like meta-commentary but are ordinary reasoning-trace register — orienting to the question is how a trace normally opens. An earlier version of this guard included them and false-flagged two of the first three generated rows, which would have blocked the merge across most of the corpus.

What the filter is actually for is model-identity leakage (`as an AI`, `I was trained by`). On cooking content that should catch close to nothing — which is the point. It is a cheap safety net, not a cleanup pass.
