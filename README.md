# Vector8 — LLM Evaluation, Routing & Prompt Optimization

A production-ready evaluation pipeline that benchmarks multiple LLMs served by
a local Ollama instance, plus two ML-engineering extensions on top of that
pipeline:

1. **Dynamic model router** — a transformer model that, given a query, predicts the model maximising `correctness / latency` so we can avoid running every model for every question.
2. **Prompt optimizer** — an iterative LLM-driven search that finds a prompt template lifting a weaker model's correctness on a held-out validation set.

---

## Repository layout

```
vector8/
├── configs/
│   └── config.yaml              # Ollama, judge, cache, API, transformer settings
├── data/
│   ├── dataset.csv              # Source Q&A dataset
│   ├── datasets/                # Frozen train/val/test splits
│   ├── models/                  # Trained router checkpoint (.pth) + tokenizer
│   └── predictions/             # Per-model and combined inference outputs
├── notebooks/
│   ├── Analysis.ipynb               # End-to-end evaluation runs + comparisons
│   ├── vector8-model-training.ipynb # Router (transformer) training & analysis
│   └── Prompt Analysis.ipynb        # Prompt optimization loop & evaluation
├── src/
│   ├── api/main.py              # FastAPI app: /evaluate, /tasks/{id}, /route
│   ├── core/                    # config loader, structured logging, request_id, versioning
│   ├── models/dataset.py        # EvaluationDataset, QueryAnswerSample, InferenceResult
│   ├── evaluation/
│   │   ├── ollama_client.py     # HTTP client w/ tenacity retry + exponential backoff
│   │   ├── atomic_function.py   # evaluate_query_answer_pair (single model × sample)
│   │   ├── multi_model_runner.py# Parallel runner: thread-per-model + cache-aware
│   │   ├── judge.py             # LLM-as-a-Judge correctness scoring
│   │   └── cache.py             # Version-keyed pickle cache + partial-hit resolver
│   ├── tasks/task_manager.py    # In-memory background task registry + thread pool
│   ├── router/                  # Transformer-based router (training + inference)
│   └── prompts/                 # Iterative prompt optimizer
├── cache/                       # Per-version pickle cache (auto-created)
├── docker-compose.yaml
├── Dockerfile
├── requirements.txt
└── main.py                      # Tiny smoke script (programmatic usage demo)
```

---

## Quick start (Docker Compose)

This brings up three services: `ollama` (model server), `ollama-init`
(one-shot model pulls — `mistral`, `llama3`, `gemma2`), and `scoring-api`
(FastAPI on `:8000`). `scoring-api` only starts after `ollama-init`
completes successfully, so the first run blocks while models download (~10–15 GB total) and subsequent runs reuse the `ollama_data` volume.

```bash
docker compose up --build
```

Verify everything is up:

```bash
docker exec ollama ollama list           # mistral, llama3, gemma2 should be listed
curl http://localhost:8000/docs          # FastAPI Swagger UI
```

To stop and clean up:

```bash
docker compose down              # keep models
docker compose down -v           # also delete the ollama_data volume
```

---

## API usage

### `POST /evaluate` — score a dataset across models

Submits an evaluation job. Returns a `task_id`; results are computed asynchronously by a background worker thread. Cached models are loaded without re-computation.

```bash
curl -X POST http://localhost:8000/evaluate \
  -H 'Content-Type: application/json' \
  -d '{
    "samples": [
      {"query": "What is a hedge?", "expected_answer": "An investment used to reduce risk on another asset."},
      {"query": "What is 7 factorial?", "expected_answer": "5040"}
    ],
    "models": ["mistral", "llama3", "gemma2"]
  }'
# -> {"task_id": "f4e2…", "status": "pending"}
```

### `GET /tasks/{task_id}` — check progress / fetch results

```bash
curl http://localhost:8000/tasks/f4e2…
# {
#   "status": "completed",
#   "models": ["mistral", "llama3", "gemma2"],
#   "result": {
#     "mistral": [{"generated_answer": "...", "latency_seconds": 1.23, "token_count": 42, "correctness": 1.0}, ...],
#     "llama3":  [...],
#     "gemma2":  [...]
#   },
#   "error": null
# }
```

Status values: `pending` → `running` → `completed` (or `failed`, with the exception in `error`). `not_found` is returned for unknown IDs.

### `POST /route` — pick the best model for a query

Returns the recommended model and the predicted `correctness / latency`
score from the trained router transformer.

```bash
curl -X POST http://localhost:8000/route \
  -H 'Content-Type: application/json' \
  -d '{"query": "Explain how an interest-rate swap works"}'
# -> {"recommended_model": "mistral", "confidence": 0.83}
```

Every response carries an `X-Request-ID` header that is propagated into the structured logs (also injected via the FastAPI middleware).

---

## Programmatic usage

The same pipeline works without HTTP — useful from notebooks or batch jobs.
See [main.py](main.py) for a runnable example:

```python
from src.models.dataset import EvaluationDataset
from src.evaluation.multi_model_runner import run_multi_model_evaluation

dataset = EvaluationDataset.from_csv("data/dataset.csv")
result  = run_multi_model_evaluation(dataset, model_names=["mistral", "llama3", "gemma2"])

for model, records in result.inferences.items():
    print(model, sum(r.correctness for r in records) / len(records))
```

`EvaluationDataset` can also be built `from_dataframe(df)`, `from_parquet(path)`,
or `from_samples([{"query": ..., "expected_answer": ...}, ...])`.

---

## Configuration

Defaults live in [configs/config.yaml](configs/config.yaml); environment
variables take precedence (loaded in [src/core/config.py](src/core/config.py)).

| YAML key                                  | Env var                  | Purpose                                   |
| ----------------------------------------- | ------------------------ | ----------------------------------------- |
| `ollama.host`                             | `OLLAMA_HOST`            | Inference endpoint, e.g. `http://ollama:11434` |
| `ollama.timeout_seconds`                  | `OLLAMA_TIMEOUT_SECONDS` | Per-request HTTP timeout                  |
| `judge.model`                             | `JUDGE_MODEL`            | Model used as LLM-as-a-Judge              |
| `judge.prompt_version`                    | `JUDGE_PROMPT_VERSION`   | Bump to invalidate cached judge scores    |
| `judge.temperature`                       | `JUDGE_TEMPERATURE`      | Judge sampling temperature (default 0.0)  |
| `judge.max_tokens`                        | `JUDGE_MAX_TOKENS`       | Cap on judge output                       |
| `cache.directory`                         | `CACHE_DIRECTORY`        | Where per-version pickles are written     |
| `api.host` / `api.port`                   | `API_HOST` / `API_PORT`  | FastAPI bind address                      |
| `transformer_model.base_transformer_model`| —                        | HF model name for the router (`roberta-base`) |
| `transformer_model.DEVICE`                | —                        | `cpu` / `cuda`                            |
| `transformer_model.max_len`               | —                        | Tokenizer max length                      |

`docker-compose.yaml` already wires `OLLAMA_HOST=http://ollama:11434` and
`JUDGE_MODEL=mistral` into the `scoring-api` container.

---

## Caching & dataset versioning

[`src/core/versioning.py`](src/core/versioning.py) computes a deterministic
SHA-256 over `(samples, sorted models, judge_model, judge_prompt_version)`,
so identical inputs always produce the same version hash. Per-model results
are pickled under `cache/<version>/<model>.pkl`
([`src/evaluation/cache.py`](src/evaluation/cache.py)).

Cache invalidates when **any** of:

- the samples change (different queries or expected answers),
- the set of models in the evaluation set changes,
- `judge.model` or `judge.prompt_version` changes.

Partial hits are supported: if 2 of 3 models already have a cache entry,
only the third is recomputed
([`resolve_cached_models`](src/evaluation/cache.py)). Concurrent writes from
the per-model worker threads are serialised with a `threading.Lock`
([`multi_model_runner.py`](src/evaluation/multi_model_runner.py)).

---

## Concurrency model

- `POST /evaluate` returns immediately with a `task_id`; the heavy work
  runs on a `ThreadPoolExecutor` owned by
  [`task_manager.py`](src/tasks/task_manager.py).
- Inside a job,
  [`run_multi_model_evaluation`](src/evaluation/multi_model_runner.py)
  spawns one thread per *missing* model so that all models execute
  concurrently. Cache hits skip the thread entirely.
- The Ollama HTTP client has tenacity-driven retries with exponential
  backoff for `Timeout`, `ConnectionError`, and `HTTPError`
  ([`ollama_client.py`](src/evaluation/ollama_client.py)).
- Logs are JSON-formatted with a per-request `request_id` propagated via
  `contextvars` ([`src/core/request_context.py`](src/core/request_context.py)
  and [`logging.py`](src/core/logging.py)).

---

## Methodology

### Train / Val / Test splits

The Train/Val/Test split is produced in
[notebooks/vector8-model-training.ipynb](notebooks/vector8-model-training.ipynb)
and stored under [data/datasets/](data/datasets/) (`train_df.csv`,
`val_df.csv`, `test_df.csv`). The same splits are reused throughout the
project — including by the prompt optimizer — to avoid leakage and keep
all reported numbers comparable across experiments.

### Analysis notebook

[notebooks/Analysis.ipynb](notebooks/Analysis.ipynb) drives the full evaluation pipeline: it calls `run_multi_model_evaluation`, captures
`generated_answer`, `latency_seconds`, `token_count`, and `correctness`
for every (sample, model) pair, and writes one DataFrame per model plus a combined one — see [data/predictions/](data/predictions/) for the materialised outputs.

### Dynamic model routing

Trained in
[notebooks/vector8-model-training.ipynb](notebooks/vector8-model-training.ipynb).

- **Architecture:** a transformer (`roberta-base` by default) fed with
  `query + [SEP] + model_name`, with two auxiliary numeric features
  (`word_count`, `char_count`) concatenated to the pooled output before a regression head.
- **Target:** `correctness / latency` for that (query, model) pair.
- **Inference:** at request time we score every candidate model and return the one with the highest predicted ratio
  ([`src/router/inference.py`](src/router/inference.py)). The score itself is returned as a confidence signal.
- **Why this objective:** correctness alone always picks the strongest model and ignores cost; latency alone picks the fastest model regardless of whether it answers correctly. The ratio captures the efficiency frontier the brief asks for.
- **Evaluation:** notebook compares the router's regret vs. an oracle (per-query best model), and against three baselines — fastest model, most-accurate model, and random selection. Analysis & conclusions live in the notebook itself.

### Prompt optimization

Implemented in
[src/prompts/prompt_optimizer.py](src/prompts/prompt_optimizer.py) and exercised in
[notebooks/Prompt Analysis.ipynb](notebooks/Prompt%20Analysis.ipynb).

Strategy: an LLM-driven iterative search over a constrained prompt space.
We start from a baseline prompt for the weaker model (`mistral`), evaluate it on a subset of `train_df`, ask the stronger model (`llama3`) to propose improvements based on observed failures, and repeat. Each candidate is
scored on `val_df`; the best-scoring prompt is then re-evaluated on `test_df` against (a) the original prompt and (b) a query-only prompt to guard against overfitting to the validation set.

```
current_prompt
      ↓
evaluate on validation set
      ↓
score correctness
      ↓
optimizer LLM analyzes failures
      ↓
generate improved prompt
      ↓
repeat (n iterations)
      ↓
final test on held-out test_df
```

**Search space** (what the optimizer is allowed to change): instruction style, reasoning strategy, output formatting, decomposition guidance, error-handling rules. **Fixed** (so improvements are attributable to prompt design alone): the `{query}` placeholder, the task definition, the model identity, the dataset, and the scoring methodology.

---

## Optional / bonus answers

- **Query-difficulty estimation.** The router predicts
  `correctness / latency` per (query, model). When the predicted score is below a threshold for the simple/faster models, we route to a stronger one; otherwise we keep the cheap model. The threshold is the lever for the difficulty/cost trade-off.
- **Cost-aware routing.** Swap the training target for either
  `correctness / latency` or `correctness − λ · token_cost` and retrain — no changes to the inference path needed.
- **Transfer learning for prompts.** Partially. Reasoning-structure improvements, instruction clarity, and formatting constraints tend to transfer across models; tokenization sensitivity, instruction-tuning differences, and model-specific biases do not. Treat transferred
  prompts as a strong starting point, not a finished product.
- **Ensemble strategies.** When the router's confidence is low we can query multiple candidates and aggregate via majority vote or have an LLM rank the outputs.
- **Online learning.** Stream production (query, model, correctness, latency) tuples back into a continually-updated router. The natural framing is a contextual bandit so the policy keeps exploring under distribution shift instead of locking in early winners.

---

## Local development (without Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# In another terminal: a local Ollama with the models pulled
ollama serve
ollama pull mistral && ollama pull llama3 && ollama pull gemma2

OLLAMA_HOST=http://localhost:11434 \
JUDGE_MODEL=mistral \
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

A minimal end-to-end smoke run is in [main.py](main.py).

---

## TODOs / improvements

- DVC for data versioning and reproducibility
- Model & dataset registry
- Train router variants for `correctness − 0.1 · latency` and
  `correctness − λ · token_cost`
- CI/CD via GitHub Actions
- Unit tests + smoke tests
- Makefile for common dev tasks
- Linting (ruff / black)
