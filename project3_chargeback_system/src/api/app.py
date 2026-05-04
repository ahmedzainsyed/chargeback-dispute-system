"""
Chargeback Prediction & Dispute API
Start: uvicorn src.api.app:app --host 0.0.0.0 --port 8002 --workers 2
"""
import time
import structlog
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import Response
from contextlib import asynccontextmanager
from prometheus_client import (
    generate_latest, CONTENT_TYPE_LATEST,
    Counter, Histogram, Gauge
)

from config.settings import settings
from src.api.schemas import ChargebackRequest, ChargebackResponse, OutcomeFeedback
from src.pipeline.processor import ChargebackProcessor
from src.feedback.loop import FeedbackCollector

log = structlog.get_logger()

CB_TOTAL      = Counter("chargebacks_total",          "Chargebacks processed", ["tier"])
DISPUTE_GEN   = Counter("dispute_responses_generated","LLM dispute letters generated")
HUMAN_QUEUE   = Counter("human_reviews_queued_total", "HIGH risk → human queue")
PROC_LAT      = Histogram("chargeback_latency_ms",    "Processing latency",
                           buckets=[100,250,500,1000,2000,5000])
QUEUE_DEPTH   = Gauge("human_review_queue_depth",     "Items in human review queue")

processor:  ChargebackProcessor = None
feedback:   FeedbackCollector   = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global processor, feedback
    log.info("starting_chargeback_service")
    processor = ChargebackProcessor()
    feedback  = FeedbackCollector(processor._db)
    log.info("service_ready", llm_enabled=settings.USE_LLM)
    yield

app = FastAPI(title="Chargeback Prediction & Dispute API",
              version="2.0.0", lifespan=lifespan)

def auth(x_api_key: str = Header(...)):
    if x_api_key != settings.API_SECRET_KEY:
        raise HTTPException(401, "Unauthorized")

@app.get("/health")
async def health():
    import redis as r
    rc = r.Redis(host=settings.REDIS_HOST, decode_responses=True)
    qd = rc.llen(settings.HUMAN_REVIEW_QUEUE_KEY)
    QUEUE_DEPTH.set(qd)
    return {
        "status": "ok",
        "llm_enabled": settings.USE_LLM,
        "human_review_queue": qd,
    }

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/v1/chargeback/process", response_model=ChargebackResponse)
async def process(req: ChargebackRequest, _=Depends(auth)):
    start  = time.perf_counter()
    result = await processor.process(req.model_dump())
    elapsed = (time.perf_counter() - start) * 1000

    PROC_LAT.observe(elapsed)
    CB_TOTAL.labels(tier=result["risk_tier"]).inc()
    if result.get("dispute_response"):
        DISPUTE_GEN.inc()
    if result.get("human_review_required"):
        HUMAN_QUEUE.inc()

    log.info("processed", cb_id=result["chargeback_id"],
             tier=result["risk_tier"], ms=round(elapsed, 1))
    return ChargebackResponse(**result)

@app.post("/v1/chargeback/outcome")
async def outcome(body: OutcomeFeedback, _=Depends(auth)):
    ok = feedback.record(body.chargeback_id, body.outcome, body.human_reviewed)
    return {"status": "recorded" if ok else "failed"}

@app.get("/v1/admin/queue")
async def queue(_=Depends(auth)):
    import redis as r
    rc = r.Redis(host=settings.REDIS_HOST, decode_responses=True)
    items = rc.lrange(settings.HUMAN_REVIEW_QUEUE_KEY, 0, 49)
    return {"depth": len(items), "items": items}
