from src.models.dataset import EvaluationDataset
from src.evaluation.multi_model_runner import run_multi_model_evaluation
from src.evaluation.judge import judge_correctness
from src.models.dataset import EvaluationDataset
from src.core.versioning import compute_dataset_version


dataset = EvaluationDataset.from_csv(path="data/dataset.csv")

dataset.samples = dataset.samples[:3]

models = ["mistral", "llama3"]

result_dataset = run_multi_model_evaluation(dataset, models)

print(result_dataset.inferences[models[0]][0])



# score = judge_correctness(
#     expected_answer="Paris",
#     generated_answer="The capital of France is Paris."
# )

# print(score)
