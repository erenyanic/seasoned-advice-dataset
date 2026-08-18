# Held-out benchmark

A 100-row evaluation set (50 English + 50 Turkish) for measuring what the
[`qwen3.5-4b-seasoned-advice-lora`](https://huggingface.co/erenyanic/qwen3.5-4b-seasoned-advice-lora)
adapter was actually fine-tuned to do. It ships as the `test_english` and `test_turkish` splits
of this dataset.

The training set answers "can the model learn this domain"; a general benchmark like Turkish
MMLU answers "did learning it break anything else". Neither answers "is the model better at
cooking advice than the base model was" — which is the question this set exists for.

## Held-out guarantee

The 50 pairs here were scraped **after** training, from the same site, with every question ID
already used in `data/sources.jsonl` excluded up front:

```bash
uv run scripts/fetch_stackexchange.py \
  --target 50 \
  --max-questions 2000 \
  --out-dir benchmark/data \
  --exclude-ids-from data/sources.jsonl
```

The `--exclude-ids-from` flag reads question IDs out of a `sources.jsonl`-style file and drops
those candidates before any filtering runs. Without it a rerun would return the *same* pairs —
the API sort is `votes`, which is deterministic — so this is what makes the split genuinely
unseen rather than nominally separate. Overlap was verified at zero after the fetch.

Same content filters as the training set: accepted answers only in this build, 150–3000
characters of answer markdown, no image-dependent pairs.

## Construction

| Stage | Script | Output |
| --- | --- | --- |
| 1. Scrape 50 fresh English pairs | `scripts/fetch_stackexchange.py` (with `--exclude-ids-from`) | `data/raw_qa.jsonl`, `data/sources.jsonl` |
| 2. Shape into the conversation schema | `scripts/build_conversations.py` | `data/test_english.jsonl` |
| 3. Translate question and answer | `benchmark/scripts/translate_qa.py` | `data/test_turkish.jsonl` |
| 4. Package into parquet splits | `scripts/package_dataset.py` | `data/test_{english,turkish}-*.parquet` |

Stages 1 and 2 reuse the training pipeline's own scripts unchanged. Stage 3 uses a benchmark-specific
script for one reason: `scripts/translate.py` translates three fields and refuses to run on any row
whose assistant turn has no reasoning trace. These rows deliberately have none, so it translates the
two fields that exist. Everything else is shared — the same `deepseek-v4-pro` client, the same
[`prompts/glossary.md`](../prompts/glossary.md) in a cached system prefix, and the same structural,
length-ratio, and English-density validation.

```bash
uv run benchmark/scripts/translate_qa.py --limit 3   # trial run, then inspect
uv run benchmark/scripts/translate_qa.py             # full 50
uv run benchmark/scripts/translate_qa.py --build     # write test_turkish.jsonl
```

## Schema

Identical to the training splits — the same five fixed keys in the same order, so the same loading
code works on both — with one deliberate difference: **`thinking` is `null` on every row.**

Reference traces would not make this benchmark more rigorous, and would arguably make it less
honest. Where a task has verifiable intermediate steps, benchmarks do ship reference reasoning
(GSM8K, MATH) because each step has a single defensible answer. Open-ended advice does not work that
way — SQuAD, MT-Bench, and AlpacaEval all score the final answer alone. There is no single correct
way to reason towards "why salt pasta water", so a generated trace here would be one plausible route
presented as a standard, which is exactly the framing this dataset's own
[limitations section](../README.md#limitations) warns against. The model still emits its `<think>`
block during evaluation; that block is stripped before its answer is compared against the human one.

Note that `thinking` keeps the arrow type `Value("string")` here rather than `Value("null")`, even
though it is null in all 100 rows. An arrow string column is nullable, so an all-null column is
valid — and declaring it `null` would give the benchmark a schema incompatible with the training
splits, which would stop all four sharing a single Hub config.
`scripts/package_dataset.py --verify` asserts both properties: that no benchmark row carries a
trace, and that all four splits report one identical schema.

## Provenance and licence

`data/sources.jsonl` carries one record per pair with question URL, answer URL, and the answer
author's display name and profile link — the same sidecar format, and the same CC BY-SA 4.0 terms,
as the training set. Attribution requirements apply here identically.

## How it is used

Scoring lives with the model, not the data:
[`Multi_Model_Benchmark_Colab.ipynb`](https://huggingface.co/erenyanic/qwen3.5-4b-seasoned-advice-lora)
runs five models across three families over these 100 questions on the same pipeline and at the
same precision, then scores them two ways — proximity to the human reference answer, and a blind,
order-randomized LLM-judge round robin on practical correctness, food-safety accuracy, actionable
specificity, and tone. Proximity is cheap and deterministic but reads only the opening of a long
answer, so it measures closeness to the corpus rather than quality; the judge carries the quality
claim. Results, and the disagreement between the two, are on the model card.
