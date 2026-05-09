from src.prompts.utils import rewrite_prompt, evaluate_prompt

def optimize_prompt(
    weak_model: str,
    validation_dataset,
    initial_prompt: str,
    iterations: int = 5,
):

    current_prompt = initial_prompt

    history = []

    best_score = -1
    best_prompt = current_prompt

    for i in range(iterations):

        score = evaluate_prompt(
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

        current_prompt = rewrite_prompt(
            current_prompt=current_prompt,
            score=score,
        )

    return {
        "best_prompt": best_prompt,
        "best_score": best_score,
        "history": history,
    }
