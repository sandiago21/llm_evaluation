Technical Assessment for ML Engineers

Objective: Build a production-ready evaluation pipeline for benchmarking LLMs via a local
inference engine.

Context
You are tasked with building a robust evaluation pipeline to benchmark multiple Large
Language Models (LLMs) available via Ollama, LMStudio, or any other local inference engine.
This pipeline must be production-ready. Specifically, you will need to:
Handle concurrency: Process multiple models in parallel using background threads
Implement caching: Avoid redundant computations by versioning and storing results
Ensure configurability: Externalize parameters via configuration files or environment
variables
Expose as a service: Provide an API layer for programmatic access to the evaluation pipeline
Most of these aspects are detailed in the requirements below. When not, we expect you to
provide how you would implement it.
Core Requirements
1. Core Function Signature & Data Structures
Atomic Function
Implement a base Python function that accepts:
A single query/expected_answer pair
A single model name is available through your inference provider
This function returns:
generated_answer : The LLM's response
latency_seconds : Time taken to generate the response (in seconds)
token_count : Number of tokens in the generated response
correctness : Boolean or score indicating correctness (via LLM-as-a-Judge)
Dataset Structure
Define a flexible data structure (class or dataclass) that supports:
A samples attribute: A collection of records, each containing at least two columns: query
and expected_answer
An inferences attribute: A dictionary keyed by model name, storing inference results for
each sample
The dataset can be initialized from multiple sources:
Ollama LMStudio

A pandas DataFrame
A path to a CSV or Parquet file
Direct instantiation with samples and (optionally) pre-existing inferences
Multi-Model Parallel Processing
For production efficiency, implement a higher-level function that:
Accepts a dataset and a list of model names (e.g., ["llama3", "mistral", "gemma2"] )
Spawns one thread per model to process all samples concurrently
Handles concurrent file/storage writes safely (use appropriate locking mechanisms)
Aggregates results into the dataset's inferences attribute
2. Dataset Versioning
Implement a versioning scheme for the dataset that captures:
A hash of the input samples (queries + expected answers)
The list of models used for inference
The judge model configuration
The version identifier should be deterministic: the same inputs must always produce the same
version hash. Use this version to:
Name output files/cache entries
Detect when re-computation is necessary
Track lineage of evaluation results
3. LLM-as-a-Judge for Correctness
Implement correctness evaluation using an LLM-as-a-judge approach
The judge model must be configurable (via YAML config or environment variable)
Design a clear prompt template for the judge to compare expected_answer vs
generated_answer
4. Caching Mechanism
Before running evaluations:
Compute the dataset version (see requirement #2)
Check if results for this version already exist in cache
If cached, load and return existing results without re-computation
Cache invalidation should occur when:

The samples change (different queries or expected answers)
A new model is added to the evaluation set
The judge model or its configuration changes
For partial cache hits (e.g., 2 of 3 models already evaluated), only compute missing models.
5. Background Task Processing
If results are not cached, launch background tasks to process each model
Multiple models should be processed in parallel (one thread per model)
Provide a mechanism to check task status (pending, running, completed, failed)
6. Containerization with Docker Compose
Provide a docker-compose.yml that orchestrates the following services:

Service   Description                        Image / Build
ollama    Ollama server for LLM inference  ollama/ollama:latest
scoring-api  Your evaluation pipeline API  Custom Dockerfile

7. Configuration Management
Externalize all configurable parameters:
Judge model name
Ollama API endpoint (default: <http://ollama:11434 > when using Docker Compose)
Cache directory/connection
Timeout settings
API host and port
Support YAML file and/or environment variables, with environment variables taking precedence.
8. Error Handling & Logging
Gracefully handle API failures (model unavailable, timeout)
Implement retry logic with exponential backoff
Use Python's logging module with structured logging (JSON format recommended for
container environments)
Include request ID tracking for API calls

ollama Ollama server for LLM

inference

ollama/ollama:
latest

scoring-api Your evaluation
pipeline API

Custom Dockerfile

Service Description Image/Build

ML Engineer Specific Work
Additional Focus: Dynamic Model Routing & Prompt Optimization
Beyond the core requirements, ML Engineer candidates must demonstrate depth in model
selection strategies and prompt optimization techniques.
1. Dynamic Model Router
Problem Statement
Given a new query and a fixed set of models (e.g., the ones used in the common base exercise),
design a system that dynamically selects the model maximizing the ratio correctness /
latency .
Requirements
Implement a router that, given a query, predicts which model to use without running all
models
The router can use any approach: query embeddings + classifier, heuristics based on query
characteristics, learned routing, etc.
Document your methodology and justify your design choices
Expose the router via an additional API endpoint:
POST /route - Given a query, return the recommended model and confidence score
Evaluation
Define metrics to assess your router's quality (e.g., regret vs. oracle selection, accuracy of
model prediction)
Provide an analysis comparing your router's selections against:
Always picking the fastest model
Always picking the most accurate model
Random selection
Report the trade-offs achieved by your router
2. Prompt Optimization for Weaker Models
Problem Statement
Assume one model (Model A) clearly outperforms others in correctness because it is inherently
better and/or the engineering team has extensively tuned its prompts. Given a different, weaker
model (Model B), develop a method to find a prompt that maximizes correctness for Model B.
Requirements

Implement an optimization approach to search for better prompts (e.g., iterative refinement,
genetic algorithms, gradient-free optimization, LLM-based prompt rewriting)
The optimization objective should be the correctness score on a validation set
Document constraints and search space (what parts of the prompt are modifiable?)
Deliverables
The optimized prompt(s) discovered
Learning curves or optimization traces showing improvement over iterations
Analysis of what prompt modifications led to the largest gains
3. Experimental Rigor
Use proper train/validation/test splits to avoid overfitting your router or prompts to the
evaluation set
Report confidence intervals or statistical significance where appropriate
Document any hyperparameters and how they were selected
Optional / Bonus
Strong candidates will identify and address:
Query difficulty estimation: Can you predict which queries are "hard" and route them to
better models?
Cost-aware routing: Extend the router to consider token costs, not just latency
Transfer learning for prompts: Do optimized prompts for Model B transfer to Model C?
Ensemble strategies: When uncertain, can you query multiple models and aggregate?
Online learning: How would the router adapt as query distributions shift over time?
Deliverables
Python source code (Git repository)
requirements.txt or pyproject.toml
Dockerfile and docker-compose.yml (functional setup)
Configuration file(s) with defaults
Analysis notebook or report covering:
Router design, implementation, and evaluation
Prompt optimization methodology and results
Comparative analysis and ablation studies
README with usage examples and methodology explanation

We encourage you to make reasonable assumptions and document them rather than getting stuck on ambiguity.
