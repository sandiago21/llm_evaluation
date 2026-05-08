"""Dataset structures and deterministic versioning."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from .config import JudgeConfig

REQUIRED_COLUMNS = ("query", "expected_answer")


@dataclass(frozen=True)
class Sample:
    query: str
    expected_answer: str


@dataclass(frozen=True)
class InferenceRecord:
    """One model's evaluation result for one sample."""

    sample_index: int
    query: str
    expected_answer: str
    model: str
    generated_answer: str
    latency_seconds: float
    token_count: int
    correctness: bool | float | None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_index": self.sample_index,
            "query": self.query,
            "expected_answer": self.expected_answer,
            "model": self.model,
            "generated_answer": self.generated_answer,
            "latency_seconds": self.latency_seconds,
            "token_count": self.token_count,
            "correctness": self.correctness,
            "error": self.error,
        }


@dataclass
class EvalDataset:
    """Holds samples + per-model inference records.

    The ``inferences`` mapping is keyed by model name and contains one
    ``InferenceRecord`` per sample (in sample order).
    """

    samples: list[Sample]
    inferences: dict[str, list[InferenceRecord]] = field(default_factory=dict)

    # ---- constructors -------------------------------------------------
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "EvalDataset":
        _validate_columns(df.columns)
        samples = [
            Sample(query=str(row["query"]), expected_answer=str(row["expected_answer"]))
            for _, row in df.iterrows()
        ]
        return cls(samples=samples)

    @classmethod
    def from_path(cls, path: str | Path) -> "EvalDataset":
        p = Path(path)
        if p.suffix.lower() == ".csv":
            df = pd.read_csv(p)
        elif p.suffix.lower() in {".parquet", ".pq"}:
            df = pd.read_parquet(p)
        else:
            raise ValueError(f"Unsupported dataset extension: {p.suffix!r}")
        return cls.from_dataframe(df)

    @classmethod
    def from_records(cls, records: Iterable[Mapping[str, Any]]) -> "EvalDataset":
        samples = [
            Sample(query=str(r["query"]), expected_answer=str(r["expected_answer"]))
            for r in records
        ]
        return cls(samples=samples)

    # ---- versioning ---------------------------------------------------
    def samples_hash(self) -> str:
        payload = json.dumps(
            [{"q": s.query, "a": s.expected_answer} for s in self.samples],
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _judge_payload(judge: JudgeConfig) -> dict[str, Any]:
        return {
            "enabled": judge.enabled,
            "provider": judge.provider,
            "model": judge.model,
            "temperature": judge.temperature,
            "prompt_template": judge.prompt_template,
        }

    def judge_hash(self, judge: JudgeConfig) -> str:
        payload = json.dumps(self._judge_payload(judge), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def model_version(self, model: str, judge: JudgeConfig) -> str:
        """Per-model cache key: depends only on samples, judge config, and the model name.

        This is what the cache uses so that adding a new model to a run does
        not invalidate previously-computed model results — the aggregate
        ``version`` may change for lineage purposes, but each model's results
        are addressable independently.
        """
        material = json.dumps(
            {"samples": self.samples_hash(), "judge": self._judge_payload(judge), "model": model},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]

    def version(self, models: Iterable[str], judge: JudgeConfig) -> str:
        """Deterministic aggregate version for (samples, models, judge config).

        Used as the lineage identifier and for the manifest filename. A new
        model in the set produces a new version, but per-model cache entries
        (keyed by ``model_version``) are still reused.
        """
        models_sorted = sorted(set(models))
        material = json.dumps(
            {
                "samples": self.samples_hash(),
                "models": models_sorted,
                "judge": self._judge_payload(judge),
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]

    # ---- summary helpers ----------------------------------------------
    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {"num_samples": len(self.samples), "models": {}}
        for model, records in self.inferences.items():
            if not records:
                out["models"][model] = {"count": 0}
                continue
            successes = [r for r in records if r.error is None]
            correct = [r for r in successes if r.correctness in (True, 1) or (isinstance(r.correctness, (int, float)) and r.correctness >= 0.5)]
            mean_latency = sum(r.latency_seconds for r in successes) / max(len(successes), 1)
            mean_tokens = sum(r.token_count for r in successes) / max(len(successes), 1)
            out["models"][model] = {
                "count": len(records),
                "errors": len(records) - len(successes),
                "accuracy": len(correct) / len(successes) if successes else None,
                "mean_latency_seconds": mean_latency,
                "mean_token_count": mean_tokens,
            }
        return out


def _validate_columns(columns: Iterable[str]) -> None:
    cols = set(columns)
    missing = [c for c in REQUIRED_COLUMNS if c not in cols]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
