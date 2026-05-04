"""End-to-end chargeback processing pipeline."""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any
import structlog, redis
import structlog

log = structlog.get_logger()

try:
    from config.settings import settings
    from src.features.engineer import single_vector
    from src.models.chargeback_model import ChargebackPredictor
    from src.models.reason_classifier import ReasonClassifier
    from src.llm.dispute_generator import DisputeGenerator
    _IMPORTS_OK = True
except Exception as e:
    log.error("import_error", error=str(e))
    _IMPORTS_OK = False


class ChargebackProcessor:
    def __init__(self):
        self.predictor  = ChargebackPredictor.load()
        self.classifier = ReasonClassifier.load()
        self.generator  = DisputeGenerator()
        self.redis      = redis.Redis(
            host=settings.REDIS_HOST, port=settings.REDIS_PORT, decode_responses=True
        )

        # Try DB (optional — graceful if no Postgres)
        self._db = None
        try:
            from sqlalchemy import create_engine
            self._db = create_engine(settings.DATABASE_URL)
        except Exception:
            log.warning("db_unavailable_running_without_persistence")

        log.info("processor_ready")

    async def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        txn_id = payload.get("transaction_id", "unknown")
        cb_id  = payload.get("chargeback_id",  "unknown")
        text   = payload.get("customer_dispute_text", "")

        # Step 1: features
        fv = single_vector(payload)

        # Step 2: risk prediction
        pred = self.predictor.predict(fv)
        score, tier = pred["score"], pred["tier"]
        log.info("scored", cb_id=cb_id, score=score, tier=tier)

        # Step 3: reason classification
        if text:
            reason_result = self.classifier.classify(text)
        else:
            reason_result = {
                "category":   payload.get("reason_category", "unrecognised"),
                "confidence": 0.5, "all_scores": {}
            }
        reason = reason_result["category"]

        # Step 4: human review queue (HIGH risk)
        if tier == "HIGH":
            self.redis.lpush(
                settings.HUMAN_REVIEW_QUEUE_KEY,
                f"{cb_id}|{score:.4f}|{reason}"
            )
            log.warning("queued_for_review", cb_id=cb_id)

        # Step 5: LLM dispute (MEDIUM + HIGH)
        dispute_response = None
        if tier in ("MEDIUM", "HIGH"):
            cb_ctx = {
                "chargeback_id": cb_id,
                "amount":        payload.get("amount"),
                "deadline_at":   (datetime.now() + timedelta(days=21)).strftime("%Y-%m-%d"),
            }
            dispute_response = await asyncio.get_event_loop().run_in_executor(
                None, self.generator.generate, payload, cb_ctx, [], reason
            )

        # Step 6: persist (if DB available)
        if self._db:
            self._store(cb_id, txn_id, score, tier, reason, dispute_response)

        return {
            "chargeback_id":        cb_id,
            "transaction_id":       txn_id,
            "risk_score":           score,
            "risk_tier":            tier,
            "reason_category":      reason,
            "reason_confidence":    reason_result["confidence"],
            "human_review_required":tier == "HIGH",
            "dispute_response":     dispute_response,
        }

    def _store(self, cb_id, txn_id, score, tier, reason, response):
        from sqlalchemy import text
        try:
            with self._db.connect() as conn:
                conn.execute(text("""
                    INSERT INTO chargebacks
                        (chargeback_id, transaction_id, risk_score, risk_tier,
                         reason_category, dispute_response, status, deadline_at)
                    VALUES
                        (:cb_id, :txn_id, :score, :tier,
                         :reason, :response, 'open', NOW() + INTERVAL '21 days')
                    ON CONFLICT (chargeback_id) DO UPDATE
                        SET risk_score=EXCLUDED.risk_score,
                            dispute_response=EXCLUDED.dispute_response
                """), dict(cb_id=cb_id, txn_id=txn_id, score=score, tier=tier,
                           reason=reason, response=response))
                conn.commit()
        except Exception as e:
            log.error("db_store_failed", error=str(e))
