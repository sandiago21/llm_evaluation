import os
import pickle
from typing import Optional, List

from src.models.dataset import (
    EvaluationDataset,
    InferenceResult,
)

from src.core.versioning import compute_dataset_version
from src.core.config import CACHE_DIRECTORY

CACHE_ROOT = CACHE_DIRECTORY

# CACHE_ROOT = "./cache"


# -------------------------
# Cache path helpers
# -------------------------

def get_cache_dir(version: str) -> str:
    """
    Cache directory per dataset version.
    """

    path = os.path.join(CACHE_ROOT, version)

    os.makedirs(path, exist_ok=True)

    return path


def get_model_cache_path(
    version: str,
    model_name: str,
) -> str:
    """
    Cache file path for a model.
    """

    cache_dir = get_cache_dir(version)

    return os.path.join(
        cache_dir,
        f"{model_name}.pkl",
    )


# -------------------------
# Save cache
# -------------------------

def save_model_cache(
    version: str,
    model_name: str,
    results: List[InferenceResult],
):
    """
    Persist inference results for one model.
    """

    cache_path = get_model_cache_path(
        version,
        model_name,
    )

    with open(cache_path, "wb") as f:
        pickle.dump(results, f)


# -------------------------
# Load cache
# -------------------------

def load_model_cache(
    version: str,
    model_name: str,
) -> Optional[List[InferenceResult]]:
    """
    Load cached results if available.
    """

    cache_path = get_model_cache_path(
        version,
        model_name,
    )

    if not os.path.exists(cache_path):
        return None

    with open(cache_path, "rb") as f:
        return pickle.load(f)


# -------------------------
# Partial cache resolver
# -------------------------

def resolve_cached_models(
    dataset: EvaluationDataset,
    model_names: List[str],
    judge_model: str,
    judge_prompt_version: str = "v1",
):
    """
    Loads cached models into dataset.inferences
    and returns missing models needing computation.
    """

    # -------------------------
    # Compute deterministic version
    # -------------------------
    version = compute_dataset_version(
        dataset=dataset,
        model_names=model_names,
        judge_model=judge_model,
        judge_prompt_version=judge_prompt_version,
    )

    missing_models = []

    # -------------------------
    # Check cache model-by-model
    # -------------------------
    for model_name in model_names:

        cached_results = load_model_cache(
            version=version,
            model_name=model_name,
        )

        if cached_results is not None:

            dataset.inferences[model_name] = cached_results

            print(f"[CACHE HIT] {model_name}")

        else:

            missing_models.append(model_name)

            print(f"[CACHE MISS] {model_name}")

    return version, missing_models
