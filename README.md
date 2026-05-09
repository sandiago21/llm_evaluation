TODO:

- DVC for data versioning
- Train Transformer model in QA format
    - Query + [SEP] + model_name and predict correctness, token_count and latency or just correctness / latency
- Optimize:
    - correctness / latency or
    - correctness - λ * token_cost

- current_prompt
      ↓
evaluate on validation set
      ↓
score correctness
      ↓
optimizer LLM analyzes failures
      ↓
generate improved prompt
      ↓
repeat



The prompt search space includes controllable components such as instruction style, reasoning strategy, output formatting, decomposition guidance, and error-handling rules. The query placeholder and task definition remain fixed to ensure evaluation consistency. Model identity, dataset, and scoring methodology are also fixed. This ensures that improvements in performance are attributable solely to prompt design.




Query difficulty estimation:
Our transformer model predicts the correctness / latency metric for each model. If the metric is quite low (below a reasonable threshold) for the simple and faster model(s) then we can route it to better models otherwise they will be routed to simpler and faster models.

Cost-aware routing:
We could easily extend the router logic and approach to optimize:
    - correctness / latency or
    - correctness - λ * token_cost

We only need to retrain the model for either metric.


Transfer learning for prompts: Do optimized prompts for Model B transfer to Model C?
Partially yes, but not fully. The reason is that why models in general share reasoning structure improvements, instruction clarity and formatting constraints, they usually do not share tokenization sensitivity, instruction tuning differences and model-specific biases.



Ensemble strategies: When uncertain, can you query multiple models and aggregate?
For uncertain routing cases, we query multiple models and aggregate outputs using voting or LLM-based ranking.




Online learning: How would the router adapt as query distributions shift over time?
We continuously update the router using logged feedback and treat routing as a contextual bandit problem under distribution shift.