from src.models.dataset import EvaluationDataset
from src.evaluation.multi_model_runner import run_multi_model_evaluation
from src.evaluation.judge import judge_correctness
from src.models.dataset import EvaluationDataset
from src.core.versioning import compute_dataset_version


# Test dataset evaluation code part
dataset = EvaluationDataset.from_samples([
    {"query": "What is AI?", "expected_answer": "Artificial Intelligence"},
    {"query": "2+2?", "expected_answer": "4"},
])

models = ["mistral"]#, "llama3", "gemma2"]

result_dataset = run_multi_model_evaluation(dataset, models)

print(result_dataset.inferences[models[0]][0].generated_answer)


# Test versioning code part
dataset = EvaluationDataset.from_samples([
    {
        "query": "What is the capital of France?",
        "expected_answer": "Paris",
    }
])

version = compute_dataset_version(
    dataset=dataset,
    model_names=["mistral", "llama3"],
    judge_model="llama3",
)

print(version)


# Test correctness scoring code part
score = judge_correctness(
    expected_answer="Paris",
    generated_answer="The capital of France is Paris."
)

print(score)
