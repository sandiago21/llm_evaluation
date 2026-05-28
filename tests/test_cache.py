import os
import pickle

from src.evaluation import cache as cache_module
from src.evaluation.cache import (
    get_cache_dir,
    get_model_cache_path,
    load_model_cache,
    resolve_cached_models,
    save_model_cache,
)
from src.models.dataset import InferenceResult


def test_get_cache_dir_creates_versioned_subdir(tmp_cache_dir):
    path = get_cache_dir("v123")
    assert os.path.isdir(path)
    assert path.endswith("v123")


def test_get_model_cache_path_is_pickle_under_version(tmp_cache_dir):
    p = get_model_cache_path("v1", "mistral")
    assert p.endswith(os.path.join("v1", "mistral.pkl"))


def test_save_and_load_model_cache_round_trip(tmp_cache_dir, inference_result):
    save_model_cache("vX", "mistral", [inference_result])
    loaded = load_model_cache("vX", "mistral")
    assert loaded is not None
    assert len(loaded) == 1
    assert loaded[0].generated_answer == inference_result.generated_answer
    assert loaded[0].correctness == inference_result.correctness


def test_load_model_cache_returns_none_when_missing(tmp_cache_dir):
    assert load_model_cache("does-not-exist", "mistral") is None


def test_save_overwrites_existing_cache(tmp_cache_dir):
    save_model_cache("v1", "mistral", [InferenceResult("a", 0.1, 1, 0.0)])
    save_model_cache("v1", "mistral", [InferenceResult("b", 0.2, 2, 1.0)])

    loaded = load_model_cache("v1", "mistral")
    assert loaded[0].generated_answer == "b"


def test_saved_cache_is_pickle_format(tmp_cache_dir, inference_result):
    save_model_cache("v1", "mistral", [inference_result])

    with open(get_model_cache_path("v1", "mistral"), "rb") as f:
        loaded = pickle.load(f)

    assert loaded[0].generated_answer == inference_result.generated_answer


def test_resolve_cached_models_full_hit(tmp_cache_dir, eval_dataset, inference_result, monkeypatch):
    # Pre-seed cache for both models using the same version function
    # the resolver uses internally.
    from src.core.versioning import compute_model_version

    for m in ["mistral", "llama3"]:
        version = compute_model_version(eval_dataset, m, "llama3")
        save_model_cache(version, m, [inference_result])

    versions, missing = resolve_cached_models(
        dataset=eval_dataset,
        model_names=["mistral", "llama3"],
        judge_model="llama3",
    )
    assert missing == []
    assert set(eval_dataset.inferences.keys()) == {"mistral", "llama3"}


def test_resolve_cached_models_partial_hit(tmp_cache_dir, eval_dataset, inference_result):
    from src.core.versioning import compute_model_version

    version_mistral = compute_model_version(eval_dataset, "mistral", "llama3")
    save_model_cache(version_mistral, "mistral", [inference_result])

    versions, missing = resolve_cached_models(
        dataset=eval_dataset,
        model_names=["mistral", "gemma2"],
        judge_model="llama3",
    )

    assert missing == ["gemma2"]
    assert "mistral" in eval_dataset.inferences
    assert "gemma2" not in eval_dataset.inferences


def test_resolve_cached_models_no_hits(tmp_cache_dir, eval_dataset):
    versions, missing = resolve_cached_models(
        dataset=eval_dataset,
        model_names=["mistral", "llama3"],
        judge_model="llama3",
    )
    assert missing == ["mistral", "llama3"]
    assert eval_dataset.inferences == {}
