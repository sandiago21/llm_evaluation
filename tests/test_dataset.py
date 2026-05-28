import pandas as pd
import pytest

from src.models.dataset import (
    EvaluationDataset,
    InferenceResult,
    QueryAnswerSample,
)


def test_from_samples_builds_dataclass_records(sample_dicts):
    ds = EvaluationDataset.from_samples(sample_dicts)
    assert len(ds.samples) == 2
    assert isinstance(ds.samples[0], QueryAnswerSample)
    assert ds.samples[0].query == "What is 2+2?"
    assert ds.samples[1].expected_answer == "Paris"
    assert ds.inferences == {}


def test_from_samples_accepts_preloaded_inferences(sample_dicts, inference_result):
    ds = EvaluationDataset.from_samples(
        sample_dicts,
        inferences={"mistral": [inference_result]},
    )
    assert ds.inferences["mistral"][0].correctness == 1.0


def test_from_dataframe_happy_path(sample_dicts):
    df = pd.DataFrame(sample_dicts)
    ds = EvaluationDataset.from_dataframe(df)
    assert ds.num_samples() == 2


def test_from_dataframe_requires_query_and_expected_answer_columns():
    df = pd.DataFrame([{"q": "x", "expected_answer": "y"}])
    with pytest.raises(ValueError, match="Missing required columns"):
        EvaluationDataset.from_dataframe(df)


def test_from_csv_round_trips_through_dataframe(tmp_path, sample_dicts):
    csv_path = tmp_path / "ds.csv"
    pd.DataFrame(sample_dicts).to_csv(csv_path, index=False)

    ds = EvaluationDataset.from_csv(str(csv_path))
    assert ds.num_samples() == 2
    assert ds.samples[0].query == sample_dicts[0]["query"]


def test_from_parquet_round_trips(tmp_path, sample_dicts):
    pq_path = tmp_path / "ds.parquet"
    pd.DataFrame(sample_dicts).to_parquet(pq_path)

    ds = EvaluationDataset.from_parquet(str(pq_path))
    assert ds.num_samples() == 2


def test_add_inference_and_is_evaluated(eval_dataset, inference_result):
    assert eval_dataset.is_evaluated("mistral") is False
    eval_dataset.add_inference("mistral", [inference_result])
    assert eval_dataset.is_evaluated("mistral") is True
    assert eval_dataset.get_models() == ["mistral"]


def test_num_samples_matches_sample_count(eval_dataset):
    assert eval_dataset.num_samples() == len(eval_dataset.samples)


def test_inference_result_fields_are_typed():
    r = InferenceResult(
        generated_answer="x",
        latency_seconds=1.0,
        token_count=2,
        correctness=0.5,
    )
    assert r.generated_answer == "x"
    assert r.latency_seconds == 1.0
    assert r.token_count == 2
    assert r.correctness == 0.5
