from src.prompts.utils import rewrite_prompt, evaluate_prompt

def optimize_prompt(
    weak_model: str,
    validation_dataset,
    initial_prompt: str,
    iterations: int = 5,
):

    current_prompt = initial_prompt

    history = []

    best_score = float("-inf")
    best_prompt = current_prompt
    best_failures = ""

    for i in range(iterations):

        score, failure_examples = evaluate_prompt(
            model_name=weak_model,
            prompt_template=current_prompt,
            dataset=validation_dataset,
        )

        history.append({
            "iteration": i,
            "score": score,
            "prompt": current_prompt,
        })

        if score > best_score:
            best_score = score
            best_prompt = current_prompt
            best_failures = failure_examples

        # Rewrite from the best prompt seen so far (not the most recent),
        # and feed the worst-graded samples back so the rewriter can target
        # specific failure modes instead of flying blind on a scalar score.
        current_prompt = rewrite_prompt(
            current_prompt=best_prompt,
            score=best_score,
            failure_examples=best_failures,
        )

    return {
        "best_prompt": best_prompt,
        "best_score": best_score,
        "history": history,
    }
