import pandas as pd
from src.evaluation.ollama_client import generate
from src.evaluation.multi_model_runner import run_multi_model_evaluation
from src.models.dataset import EvaluationDataset

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
    n_failure_examples: int = 5,
):
    """
    Evaluate ``prompt_template`` on ``dataset`` against ``model_name``.

    Returns ``(score, failure_examples)`` where ``score`` is mean
    correctness on the validation set and ``failure_examples`` is a
    formatted string with the worst-graded samples (for the optimizer
    to feed back into the rewriter).

    The input ``dataset`` is **not** mutated — a fresh templated
    dataset is constructed for the evaluation run.
    """

    # Build a templated copy without touching the input dataset.
    templated_dataset = EvaluationDataset.from_samples(
        [
            {
                "query": prompt_template.format(query=s.query),
                "expected_answer": s.expected_answer,
            }
            for s in dataset.samples
        ]
    )

    result_dataset = run_multi_model_evaluation(templated_dataset, [model_name])
    inferences = result_dataset.inferences[model_name]

    rows = [
        {
            "query": dataset.samples[i].query,
            "expected_answer": dataset.samples[i].expected_answer,
            "generated_answer": inf.generated_answer,
            "correctness": inf.correctness,
        }
        for i, inf in enumerate(inferences)
    ]
    df = pd.DataFrame(rows)

    score = df["correctness"].mean()

    # Worst-graded samples become the failure-example signal sent to the
    # rewriter. Tie-break by shorter generated answer (often a refusal).
    worst = df.sort_values(
        ["correctness", "generated_answer"],
        ascending=[True, True],
    ).head(n_failure_examples)

    failure_examples = "\n\n".join(
        f"Query: {row['query']}\n"
        f"Expected: {row['expected_answer']}\n"
        f"Generated: {row['generated_answer']}\n"
        f"Correctness: {row['correctness']}"
        for _, row in worst.iterrows()
    )

    return score, failure_examples
