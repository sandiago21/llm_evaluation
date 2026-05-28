import os

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List

from src.models.dataset import EvaluationDataset
from src.tasks.task_manager import (
    launch_evaluation_task,
    get_task_status,
)

from src.core.request_context import (
    generate_request_id,
    set_request_id,
)

from src.core.logging import logger


SKIP_ROUTER_MODEL = os.getenv("SKIP_ROUTER_MODEL", "").lower() in {"1", "true", "yes"}


app = FastAPI()


@app.on_event("startup")
def startup():
    # Router imports trigger a HuggingFace tokenizer download and a torch.load
    # of a checkpoint that isn't tracked in git. When SKIP_ROUTER_MODEL is set
    # (CI smoke tests, fresh clones without the .pth file) we skip both.
    if SKIP_ROUTER_MODEL:
        logger.info("SKIP_ROUTER_MODEL set — /route will return 503")
        return

    from src.router.model import load_router_model

    load_router_model()
    logger.info("Router model loaded successfully")

# -------------------------
# Middleware
# -------------------------

@app.middleware("http")
async def add_request_id(
    request: Request,
    call_next,
):

    request_id = generate_request_id()

    set_request_id(request_id)

    logger.info(
        "Incoming request",
        extra={
            "request_id": request_id,
        },
    )

    response = await call_next(request)

    response.headers["X-Request-ID"] = request_id

    return response


# -------------------------
# Request schemas
# -------------------------

class SampleRequest(BaseModel):
    query: str
    expected_answer: str


class EvaluationRequest(BaseModel):
    samples: List[SampleRequest]
    models: List[str]


class RouteRequest(BaseModel):
    query: str


# -------------------------
# Routes
# -------------------------

@app.post("/evaluate")
def evaluate(request: EvaluationRequest):

    dataset = EvaluationDataset.from_samples(
        [
            {
                "query": s.query,
                "expected_answer": s.expected_answer,
            }
            for s in request.samples
        ]
    )

    task_id = launch_evaluation_task(
        dataset=dataset,
        model_names=request.models,
    )

    return {
        "task_id": task_id,
        "status": "pending",
    }


@app.get("/tasks/{task_id}")
def task_status(task_id: str):

    return get_task_status(task_id)


@app.post("/route")
def route_query(request: RouteRequest):

    if SKIP_ROUTER_MODEL:
        raise HTTPException(
            status_code=503,
            detail="Router model disabled via SKIP_ROUTER_MODEL",
        )

    from src.router.inference import predict_best_model

    return predict_best_model(query=request.query)
