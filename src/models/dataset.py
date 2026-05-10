from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import pandas as pd
import os


# -------------------------
# Core data structures
# -------------------------

@dataclass
class QueryAnswerSample:
    query: str
    expected_answer: str


@dataclass
class InferenceResult:
    generated_answer: str
    latency_seconds: float
    token_count: int
    correctness: float


# -------------------------
# Main Dataset class
# -------------------------

@dataclass
class EvaluationDataset:
    """
    Central dataset abstraction for LLM evaluation.
    """

    samples: List[QueryAnswerSample]
    inferences: Dict[str, List[InferenceResult]] = field(default_factory=dict)

    # -------------------------
    # Factory methods
    # -------------------------

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "EvaluationDataset":
        """
        Initialize dataset from pandas DataFrame.
        Expected columns: query, expected_answer
        """

        required_cols = {"query", "expected_answer"}

        if not required_cols.issubset(df.columns):
            raise ValueError(f"Missing required columns: {required_cols}")

        samples = [
            QueryAnswerSample(
                query=row["query"],
                expected_answer=row["expected_answer"],
            )
            for _, row in df.iterrows()
        ]

        return cls(samples=samples)

    @classmethod
    def from_csv(cls, path: str) -> "EvaluationDataset":
        """
        Load dataset from CSV file.
        """
        df = pd.read_csv(path)
        return cls.from_dataframe(df)

    @classmethod
    def from_parquet(cls, path: str) -> "EvaluationDataset":
        """
        Load dataset from Parquet file.
        """
        df = pd.read_parquet(path)
        return cls.from_dataframe(df)

    @classmethod
    def from_samples(
        cls,
        samples: List[Dict[str, str]],
        inferences: Optional[Dict[str, List[InferenceResult]]] = None,
    ) -> "EvaluationDataset":
        """
        Direct instantiation from raw dict samples.
        """

        parsed_samples = [
            QueryAnswerSample(
                query=s["query"],
                expected_answer=s["expected_answer"],
            )
            for s in samples
        ]

        return cls(
            samples=parsed_samples,
            inferences=inferences or {},
        )

    # -------------------------
    # Utility methods
    # -------------------------

    def add_inference(
        self,
        model_name: str,
        results: List[InferenceResult],
    ) -> None:
        """
        Store results for a specific model.
        """
        self.inferences[model_name] = results

    def get_models(self) -> List[str]:
        """
        Return evaluated model names.
        """
        return list(self.inferences.keys())

    def is_evaluated(self, model_name: str) -> bool:
        """
        Check if model already has results.
        """
        return model_name in self.inferences

    def num_samples(self) -> int:
        return len(self.samples)
