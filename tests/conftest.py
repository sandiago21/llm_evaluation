import os
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# src/core/config.py opens "configs/config.yaml" with a relative path at
# import time, so the test process must run with REPO_ROOT as its CWD.
os.chdir(REPO_ROOT)


@pytest.fixture
def sample_dicts():
    return [
        {"query": "What is 2+2?", "expected_answer": "4"},
        {"query": "Capital of France?", "expected_answer": "Paris"},
    ]


@pytest.fixture
def eval_dataset(sample_dicts):
    from src.models.dataset import EvaluationDataset

    return EvaluationDataset.from_samples(sample_dicts)


@pytest.fixture
def inference_result():
    from src.models.dataset import InferenceResult

    return InferenceResult(
        generated_answer="4",
        latency_seconds=0.5,
        token_count=3,
        correctness=1.0,
    )


@pytest.fixture
def tmp_cache_dir(tmp_path, monkeypatch):
    """Redirect the cache module to write into a temp directory."""
    from src.evaluation import cache as cache_module

    monkeypatch.setattr(cache_module, "CACHE_ROOT", str(tmp_path))
    return tmp_path
