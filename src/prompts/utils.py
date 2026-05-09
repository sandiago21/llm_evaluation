import pandas as pd
from src.evaluation.ollama_client import generate
from src.evaluation.multi_model_runner import run_multi_model_evaluation

MODEL = "llama3"

OPTIMIZER_SYSTEM_PROMPT = """
You are an expert prompt engineer.

Your job is to improve prompts used for a weaker LLM.

You must:
- Increase factual correctness
- Improve reasoning quality
- Reduce hallucinations
- Make instructions clearer
- Keep prompt concise (do not over-expand)

Return ONLY the improved prompt.
"""



def rewrite_prompt(
    current_prompt: str,
    score: float,
    failure_examples: str = None,
) -> str:
    """
    Uses a strong LLM to improve a weak-model prompt.
    """

    user_message = f"""
Current prompt:
----------------
{current_prompt}

Validation score:
----------------
{score}

"""

    if failure_examples:
        user_message += f"""
Failure cases:
--------------
{failure_examples}
"""

    user_message += """

Rewrite this prompt to improve performance of the weak model.
Return ONLY the improved prompt. No explanations.
"""

    response = generate(
        model_name=MODEL,
        prompt=OPTIMIZER_SYSTEM_PROMPT + "\n\n" + user_message,
    )

    improved_prompt = response["response"].strip()

    return improved_prompt


def evaluate_prompt(
    model_name, prompt_template, dataset,
):
    for i in range(len(dataset.samples)):
        prompt = prompt_template.format(query=dataset.samples[i].query)
        dataset.samples[i].query = prompt

    models_preds = {
        model_name: {
            "generated_answer": [],
            "latency_seconds": [],
            "token_count": [],
            "correctness": [],
        }
    }

    result_dataset = run_multi_model_evaluation(dataset, [model_name])
    model_inference_results = result_dataset.inferences[model_name]

    for model_inference_result in model_inference_results:
        models_preds[model_name]["generated_answer"].append(model_inference_result.generated_answer)
        models_preds[model_name]["latency_seconds"].append(model_inference_result.latency_seconds)
        models_preds[model_name]["token_count"].append(model_inference_result.token_count)
        models_preds[model_name]["correctness"].append(model_inference_result.correctness)
    
    model_preds_df = pd.DataFrame(models_preds[model_name])
    model_preds_df["correctness_over_latency"] = model_preds_df["correctness"] / model_preds_df["latency_seconds"]

    score = model_preds_df["correctness_over_latency"].mean()

    return score
