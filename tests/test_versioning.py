from src.core.versioning import (
    compute_dataset_version,
    compute_model_version,
)
from src.models.dataset import EvaluationDataset


def _ds(samples):
    return EvaluationDataset.from_samples(samples)


def test_dataset_version_deterministic(sample_dicts):
    v1 = compute_dataset_version(_ds(sample_dicts), ["mistral", "llama3"], "llama3")
    v2 = compute_dataset_version(_ds(sample_dicts), ["mistral", "llama3"], "llama3")
    assert v1 == v2


def test_dataset_version_independent_of_model_order(sample_dicts):
    v1 = compute_dataset_version(_ds(sample_dicts), ["mistral", "llama3"], "llama3")
    v2 = compute_dataset_version(_ds(sample_dicts), ["llama3", "mistral"], "llama3")
    assert v1 == v2


def test_dataset_version_changes_with_samples(sample_dicts):
    v1 = compute_dataset_version(_ds(sample_dicts), ["mistral"], "llama3")
    altered = sample_dicts + [{"query": "new", "expected_answer": "x"}]
    v2 = compute_dataset_version(_ds(altered), ["mistral"], "llama3")
    assert v1 != v2


def test_dataset_version_changes_with_model_set(sample_dicts):
    v1 = compute_dataset_version(_ds(sample_dicts), ["mistral"], "llama3")
    v2 = compute_dataset_version(_ds(sample_dicts), ["mistral", "gemma2"], "llama3")
    assert v1 != v2


def test_dataset_version_changes_with_judge_model(sample_dicts):
    v1 = compute_dataset_version(_ds(sample_dicts), ["mistral"], "llama3")
    v2 = compute_dataset_version(_ds(sample_dicts), ["mistral"], "mistral")
    assert v1 != v2


def test_dataset_version_changes_with_prompt_version(sample_dicts):
    v1 = compute_dataset_version(_ds(sample_dicts), ["mistral"], "llama3", "v1")
    v2 = compute_dataset_version(_ds(sample_dicts), ["mistral"], "llama3", "v2")
    assert v1 != v2


def test_model_version_independent_of_other_models(sample_dicts):
    """Per-model key must not depend on what other models are in the run."""
    v_alone = compute_model_version(_ds(sample_dicts), "mistral", "llama3")
    v_with_others = compute_model_version(_ds(sample_dicts), "mistral", "llama3")
    assert v_alone == v_with_others


def test_model_version_differs_by_model_name(sample_dicts):
    v_a = compute_model_version(_ds(sample_dicts), "mistral", "llama3")
    v_b = compute_model_version(_ds(sample_dicts), "gemma2", "llama3")
    assert v_a != v_b


def test_model_version_is_sha256_hex():
    ds = _ds([{"query": "q", "expected_answer": "a"}])
    v = compute_model_version(ds, "mistral", "llama3")
    assert len(v) == 64
    int(v, 16)
