---
language:
  - tr
  - en
license: cc-by-sa-4.0
task_categories:
  - text-generation
  - question-answering
tags:
  - cooking
  - food-science
  - instruction-tuning
  - reasoning
  - chain-of-thought
  - turkish
size_categories:
  - n<1K
configs:
  - config_name: default
    data_files:
      - split: english
        path: data/english-*.parquet
      - split: turkish
        path: data/turkish-*.parquet
---

# Seasoned Advice Dataset

A bilingual (Turkish / English) instruction-tuning dataset of **500 conversations** in the cooking and food-science domain, derived from real human question/answer pairs on [Seasoned Advice](https://cooking.stackexchange.com), the Stack Exchange site for cooking.

Every question and answer is human-written content collected through the Stack Exchange API — none of it is model-generated. Each assistant turn additionally carries a **reasoning trace** in a dedicated `thinking` field, making the dataset suitable for training or evaluating models that expose intermediate reasoning.

|              |                                                                    |
| ------------ | ------------------------------------------------------------------ |
| **Examples** | 500 per split                                                      |
| **Splits**   | `english`, `turkish` (parallel — row *n* is the same conversation) |
| **Turns**    | 2 (one user, one assistant)                                        |
| **Domain**   | Cooking, kitchen technique, food safety, food science              |
| **Licence**  | CC BY-SA 4.0                                                       |

## Intended use

Supervised fine-tuning and evaluation of instruction-following models, with two particular fits:

- **Turkish-language SFT.** Turkish is under-represented in openly licensed instruction data, and the domain content here is practical rather than synthetic.
- **Reasoning-trace training.** Every assistant turn pairs a final answer with the reasoning that leads to it, in both languages.

Because the splits are parallel, the dataset also supports translation evaluation and cross-lingual consistency work.

## Format

Each example is a two-message record in the **conversational** (or "messages") format — a list of message objects carrying `role` and `content`, the shape used by OpenAI-style chat APIs and by most fine-tuning stacks. Three optional fields extend it: `thinking`, carrying the assistant's reasoning separately from its user-facing `content`; and `images` / `tool_calls` from the multimodal and tool-calling conventions, which stay `null` throughout.

```json
[
  {
    "content": "Makarna suyuna neden tuz eklenir?",
    "images": null,
    "role": "user",
    "thinking": null,
    "tool_calls": null
  },
  {
    "content": "Tuz lezzet katar, ancak aynı zamanda ...",
    "images": null,
    "role": "assistant",
    "thinking": "Kullanıcı makarna suyuna tuz eklemenin etkisini soruyor ...",
    "tool_calls": null
  }
]
```

The five keys are fixed and no extra keys are added, so the data loads directly with standard chat templates. Per-example provenance is kept in a separate sidecar file rather than inside the records — see [Attribution](#attribution).

### Loading

```python
from datasets import load_dataset

ds = load_dataset("Erenyanic/seasoned-advice-dataset")
ds["turkish"][0]["train"]   # -> list of two message dicts
```

Working files are JSONL (one conversation per line, human-readable and diffable). The release is parquet, a compressed columnar format — not text, which is why a plain editor shows binary. Read it with `datasets`, `pandas`, or any parquet viewer.

### Schema

Two splits, each a single column named `train` holding the message list:

```python
DatasetDict({
    turkish: Dataset({features: ['train'], num_rows: 500})
    english: Dataset({features: ['train'], num_rows: 500})
})

{'train': List({
    'content':    Value('string'),
    'images':     Value('null'),
    'role':       Value('string'),
    'thinking':   Value('string'),
    'tool_calls': Value('null'),
})}
```

`images` and `tool_calls` carry the arrow type `null` rather than a nullable string, which follows from those fields being `None` in every row.

## Construction

| Stage                          | Script                           | Output                                    |
| ------------------------------ | -------------------------------- | ----------------------------------------- |
| 1. Scrape English pairs        | `scripts/fetch_stackexchange.py` | `data/raw_qa.jsonl`, `data/sources.jsonl` |
| 2. Shape into schema           | `scripts/build_conversations.py` | `data/english.jsonl`                      |
| 3. Generate English `thinking` | `scripts/generate_thinking.py`   | fills `thinking` in `data/english.jsonl`  |
| 4. Translate all three fields  | `scripts/translate.py`           | `data/turkish.jsonl`                      |
| 5. Package into parquet splits | `scripts/package_dataset.py`     | `data/{english,turkish}-*.parquet`        |

The conversation schema is fixed at stage 2, before any content is generated, so later stages only ever rewrite field *values* and never the structure.

**Reasoning is generated in English first, then translated with the Q&A.** The reverse order — translating first, then generating Turkish reasoning — would produce two splits whose reasoning diverges. Generating once and translating keeps the splits genuinely parallel and keeps the reasoning close to the source material it was derived from.

### Stage 1 — scraping

```bash
uv run scripts/fetch_stackexchange.py --target 500
```

Questions are pulled from the Stack Exchange API sorted by score (`--sort votes`), which surfaces the site's canonical, best-answered questions and makes runs reproducible. `activity`, `creation`, and `hot` are also accepted. An answer is kept only if:

- it is the **accepted** answer, or otherwise scores **>= 3**
- its markdown length is between **150 and 3000 characters**
- neither the question nor the answer embeds an image

The image rule is deliberate: this is a text-only dataset, so any pair whose meaning depends on a picture is dropped rather than silently degraded. No images are downloaded at any point.

Roughly 1500 candidate questions are pulled to yield 500 survivors. In the current build all 500 came from accepted answers, so the `score >= 3` fallback never triggered.

Bodies are converted from HTML to markdown with backslash-escaping disabled, and question titles are HTML-unescaped — the API returns titles encoded, and without this step 15% of them carried raw `&quot;` / `&#39;` entities into the text.

### Stage 2 — schema

```bash
uv run scripts/build_conversations.py --split english
```

The user message is the question title joined to its body; the assistant message is the answer. The script asserts message count, role order, exact key set, and non-empty content on every row.

### Stages 3 and 4 — reasoning and translation

Both stages are one API call per example against `deepseek-v4-pro` with thinking mode enabled at `reasoning_effort: medium`. Prompt text is mirrored in [`prompts/`](prompts/) for reading; the scripts embed their own copy so a run is never split across two sources of truth.

Generation totals across both stages, including trial runs, regenerations, and repairs:

| **Cost**         | $3.26 USD |
| ---------------- | --------- |
| **API requests** | 1,299     |
| **Tokens**       | 5,957,638 |

The request count exceeds 1,000 because rows failing validation are regenerated rather than accepted. Cost stayed low because the glossary sits in an identical system prefix on every call, so DeepSeek's automatic prefix cache served the bulk of input tokens at a fraction of the uncached rate.

```bash
uv run scripts/generate_thinking.py --limit 20   # trial run, then inspect
uv run scripts/generate_thinking.py              # full 500
uv run scripts/generate_thinking.py --merge      # fold traces into english.jsonl

uv run scripts/translate.py --limit 20
uv run scripts/translate.py
uv run scripts/translate.py --build              # write turkish.jsonl
```

The API key is read from `DEEPSEEK_API` in the environment or `.env` (gitignored). Both scripts are **resumable**: results append to a checkpoint keyed by row index, so an interrupted run continues rather than restarting, and failed rows are retried by re-running.

[`prompts/glossary.md`](prompts/glossary.md) locks the EN→TR rendering of every cooking term frequent in the corpus. It sits in the system message, identical on every call, so DeepSeek's automatic prefix cache serves it at a fraction of the price of a miss. Without it the same term drifts across 500 examples.

## Quality controls

Reasoning traces are generated, so they get screened rather than trusted.

**The trace is taken from `content`, not `reasoning_content`.** The API returns both: `content` is the trace the model deliberately wrote; `reasoning_content` is its internal deliberation *about how to write one*. The second is meta-commentary about the task and is discarded.

**Identity-leakage filter.** Traces are scanned for model-identity terms (`as an AI`, `language model`, vendor names). Stage 3 refuses to merge while any row is flagged. Deliberately *not* filtered: phrases like "the user is asking", which read as meta-commentary but are ordinary reasoning-trace register.

**Answer-contamination detector.** Some traces finish reasoning and then go on to write the answer itself — one early sample appended a 1,555-character near-copy of the source answer. Detection uses the longest verbatim run shared between trace and answer, thresholded at 100 characters. That threshold comes from the measured distribution: clean traces share 16–39 characters (incidental phrases like "the boiling point of water"), contaminated ones share hundreds. Flagged rows are regenerated. In the full run this caught **8.8%** of traces.

A rejected alternative is documented here because it looked convincing: flagging prose that continues past a "structure the response" marker. It does not separate the classes — clean traces reach 600+ characters of legitimate planning while a contaminated one sat at 527. Used as a filter it would have flagged 55% of rows, almost all of them fine.

**Translation structure check.** Stage 4 compares markdown marker counts (headings, list items, code fences, links) between source and translation and warns on divergence, which catches reformatting.

## Provenance

| Component                  | Source                                                          |
| -------------------------- | --------------------------------------------------------------- |
| Question text (English)    | Human-written, scraped from Seasoned Advice                     |
| Answer text (English)      | Human-written, scraped from Seasoned Advice                     |
| `thinking` (English)       | Generated by `deepseek-v4-pro`, conditioned on the scraped pair |
| Turkish split (all fields) | `deepseek-v4-pro` translation of the English split              |

## Limitations

- **Reasoning traces are reconstructions.** Forum answers carry no chain of thought. Traces are generated *from* the finished answer, so they are plausible routes to a known conclusion, not records of how the original author reasoned.
- **The Turkish split is machine-translated.** Terminology is controlled by a reviewed glossary, and every row is validated for structure, relative length, and language before acceptance — rows failing any check were regenerated. The text has not been fully post-edited by a human translator.
- **Domain and register are narrow.** Content reflects the conventions of one Stack Exchange community: largely Anglophone home cooking, with the measurement systems and ingredient availability that implies.
- **Answers reflect community consensus, not authority.** High-scoring Stack Exchange answers are well-regarded, not peer-reviewed. Food-safety claims in particular should not be treated as authoritative.

## Attribution

Source content is from Stack Exchange's Seasoned Advice, licensed **[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)**.

`data/sources.jsonl` carries one record per example with the question URL, the answer URL, and the answer author's display name and profile link:

```json
{
  "index": 0,
  "question_url": "https://cooking.stackexchange.com/q/11903",
  "answer_url": "https://cooking.stackexchange.com/a/12160",
  "answer_author": "Josh",
  "answer_author_url": "https://cooking.stackexchange.com/users/3345/josh",
  "license": "CC BY-SA 4.0"
}
```

Attribution lives in a sidecar rather than inside the conversation records so the training schema stays clean. CC BY-SA requires attributing authors, not merely linking the source site — hence the per-answer author fields.

### Licence of this dataset

CC BY-SA is a share-alike licence, so this derived dataset is also released under **CC BY-SA 4.0**. Translations and reasoning traces are derivative works of the original posts and inherit the same terms.

## Files

```text
seasoned-advice-dataset/
├── README.md
├── scripts/
│   ├── fetch_stackexchange.py    stage 1: Stack Exchange API scraper
│   ├── build_conversations.py    stage 2: shape into the target schema
│   ├── _deepseek.py              shared DeepSeek client (retry, usage, caching)
│   ├── generate_thinking.py      stage 3: English reasoning traces
│   └── translate.py              stage 4: Turkish translation
├── prompts/
│   ├── glossary.md               EN->TR cooking terminology lock
│   ├── thinking_en.md            stage 3 prompt, for reading
│   └── translate_tr.md           stage 4 prompt, for reading
└── data/
    ├── raw_qa.jsonl              500 English pairs with scrape metadata
    ├── english.jsonl             500 conversations, English split
    ├── turkish.jsonl             500 conversations, Turkish split
    └── sources.jsonl             500 attribution records
```
