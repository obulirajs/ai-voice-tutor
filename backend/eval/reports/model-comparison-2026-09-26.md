# Local Ollama model comparison — 2026-09-26

Benchmarks the three local Ollama models against the full grounded-chat
pipeline (`app.orchestration.handle_turn`, the same code path as
`POST /chat`) using `backend/scripts/compare_tutor_models.py` and the
42-item eval set at `backend/eval/tutor_eval_set.json`. Purpose: pick the
best Ollama default for offline/dev use (`OLLAMA_MODEL`) — this does not
change the live-conversation default, which stays the Anthropic API per
`docs/technical-design.md`'s hardware assessment (Raj's laptop has no GPU
capable of fast local inference).

Each model ran a warm-up turn (discarded), then all 42 items at
`temperature=0.0`, mode=`ORCHESTRATION_DEFAULTS.default_teaching_mode`,
against the real SQLite-backed vector store. Raw output: `eval/logs/*.log`
(console) and `eval/results/*.jsonl` (per-item records).

## Summary

| Model | Processor | Errors | Refusal-classification correct | Avg latency | p50 | p95 | `<think>` leaks | Avg reply length |
|---|---|---|---|---|---|---|---|---|
| llama3.2:1b | 100% GPU | 2/42 (timeouts) | 24/40 (60.0%) | 8.7s | 5.6s | 23.6s | 0 | 70.4 words |
| qwen2.5:3b | 100% GPU | 3/42 (2 timeouts, 1 connection error) | 31/39 (79.5%) | 48.5s | 46.8s | 109.4s | 0 | 35.9 words |
| qwen3.5:2b | 33%/67% CPU/GPU | 0/42 | 36/42 (85.7%) | 58.8s | 44.7s | 126.6s | 0 | 92.3 words |

"Refusal-classification correct" = the item's `guardrail_refused` outcome
(no sources retrieved, or the model emitting the canned not-covered
sentence) matched the eval set's `expected_refusal` label, out of the
items that didn't error out.

## Correctness detail

**llama3.2:1b — 16 wrong refusal calls, worst of the three.** Over-refuses:
12 of its 16 misses are legitimately-answerable questions
(vocab/grammar/ocr-page/comprehension/table-or-diagram) that it refused
anyway, plus it also under-refuses on 3 of the 6 dedicated `refusal-*`
adversarial items and 1 `out-of-scope` item (answered when it should have
declined). It also timed out on 2 items (`fr-golden-17`, `fr-golden-20`).

**qwen2.5:3b — 8 wrong, mid-pack.** Same over-refusal pattern on 7
answerable items (mostly `ocr-page`/vocab/grammar/comprehension), plus one
missed adversarial refusal (`refusal-06`). 3 errors: 2 timeouts
(`fr-golden-18`, `fr-golden-23`) and one `ResponseError` on `fr-golden-22`
from a closed connection to the local Ollama server mid-call — likely
transient (server load from the long 3b run), not a model defect.

**qwen3.5:2b — 6 wrong, best of the three, and zero errors.** Still
over-refuses on the same recurring 4 items (`fr-golden-02`, `-07`, `-12`,
`-21` — these trip all three models, worth checking whether their source
passages are borderline in the retrieval index rather than blaming the
models), plus misses one adversarial item (`fr-golden-25`) and one
`french-language` item (`fr-lang-01`, over-refused).

**Recurring false-refusals across all 3 models:** `fr-golden-02`,
`fr-golden-07`, `fr-golden-12`, `fr-golden-21` were refused by every model
tested. Since this reproduces regardless of model choice, it points at
retrieval (chunking/scoring not surfacing the right passage, so the
guardrail's "no sources" or the model's own "related but doesn't answer"
path fires) rather than at the LLM — worth a follow-up look at
`app/knowledge/retrieval.py` for these four questions specifically, not a
model-comparison fix.

## Latency and hardware fit

llama3.2:1b is ~5-7x faster than the other two and fully GPU-resident,
consistent with `docs/technical-design.md`'s hardware note (MX150, ~2GB
VRAM) — it's the only one of the three that fits entirely on-GPU here.
qwen2.5:3b and qwen3.5:2b both run far slower (p95 well over 100s per
turn), and qwen3.5:2b partially spills to CPU (33%/67% split), which is
this machine confirming the doc's prediction that non-trivial local models
won't feel conversational on this hardware.

## Recommendation

None of the three is a clean win: llama3.2:1b is fast enough to be usable
for offline/dev but is the least accurate (60% refusal-classification,
worst over-refusal rate); qwen3.5:2b is the most accurate but at
40-130s/turn is too slow to use interactively even for dev/offline
purposes. Given this is only the offline/dev fallback (Anthropic stays the
live default), lean toward keeping `llama3.2:1b` as `OLLAMA_MODEL` for
day-to-day dev speed, but treat its answers as unreliable for grading
grounding quality — use qwen3.5:2b (accepting the latency) when the dev
task specifically needs to sanity-check retrieval/grounding behavior
offline. Separately, the 4 recurring false-refusals are worth root-causing
in retrieval before drawing further conclusions from this eval set, since
they cap every model's accuracy ceiling here at ~90%.

## Raw data

- `eval/results/llama3.2-1b.jsonl`, `eval/results/qwen2.5-3b.jsonl`, `eval/results/qwen3.5-2b.jsonl` — per-item records (`type: "item_result"`) plus a `run_header` record with the ollama version/processor/timestamp.
- `eval/logs/*.log` — console output from each run.
