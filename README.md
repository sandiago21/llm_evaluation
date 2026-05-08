Skeleton is complete. Here's what got built and verified:

Layout

pyproject.toml, Dockerfile, docker-compose.yml, .dockerignore, .gitignore
config/default.yaml — every setting documented; env vars (EVAL_*, __ for nesting) override YAML
src/eval_pipeline/
config.py — Pydantic settings with custom YAML source so env > YAML
logging_config.py — JSON logs + per-request request_id contextvar
dataset.py — EvalDataset from CSV/Parquet/DataFrame/records; deterministic samples_hash, model_version, version
providers/ — InferenceProvider Protocol, OllamaProvider (httpx + tenacity exp-backoff retries), deterministic MockProvider
judge.py — LLM-as-Judge with configurable prompt template
inference.py — atomic evaluate_one(...) -> InferenceRecord (latency/tokens/correctness/error)
cache.py — per-model Parquet keyed by (samples, judge, model) for partial hits; manifest under versions/<version>/manifest.json for lineage; per-path threading.Lock
tasks.py — in-memory task registry (pending/running/completed/failed)
runner.py — ThreadPoolExecutor one thread per model, inner pool per sample
api.py — FastAPI: /health, /evaluate (sync or async), /tasks/{id}, /results/{version}, /route (501 placeholder)
tests/test_smoke.py — 5 tests covering versioning determinism, full run, partial cache hit, API flow, cache round-trip
Verified

All 5 pytest tests pass with no warnings.
Live uvicorn server: /health honors EVAL_PROVIDER=mock, sync + async /evaluate both return correct per-model summaries, /tasks/{id} polling works, /route returns 501.
Reserved extension points (not implemented per scope)

/route endpoint stub for the dynamic model router
No prompt-optimization module yet
No analysis notebook
Run it:

Tests: .venv/bin/python -m pytest
Local API (mock provider): EVAL_PROVIDER=mock EVAL_JUDGE__PROVIDER=mock .venv/bin/uvicorn eval_pipeline.api:app
Full stack: docker compose up --build (then ollama pull llama3.2 etc. against the running ollama service)
