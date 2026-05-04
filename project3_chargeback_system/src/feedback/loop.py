"""Feedback loop: record bank outcomes → triggers retraining."""
import structlog
from typing import Optional

log = structlog.get_logger()

class FeedbackCollector:
    def __init__(self, db_engine=None):
        self._db = db_engine

    def record(self, chargeback_id: str, outcome: int,
               human_reviewed: bool = False) -> bool:
        if not self._db:
            log.info("feedback_no_db", cb_id=chargeback_id, outcome=outcome)
            return True
        from sqlalchemy import text
        try:
            with self._db.connect() as conn:
                conn.execute(text("""
                    UPDATE chargebacks
                    SET outcome_label  = :outcome,
                        status         = CASE WHEN :outcome=1 THEN 'won' ELSE 'lost' END,
                        human_reviewed = :hr
                    WHERE chargeback_id = :cb_id
                """), dict(outcome=outcome, hr=human_reviewed, cb_id=chargeback_id))
                conn.commit()
            log.info("outcome_recorded", cb_id=chargeback_id, outcome=outcome)
            self._check_retrain()
            return True
        except Exception as e:
            log.error("feedback_failed", error=str(e))
            return False

    def _check_retrain(self):
        from sqlalchemy import text
        try:
            with self._db.connect() as conn:
                n = conn.execute(text("""
                    SELECT COUNT(*) FROM chargebacks
                    WHERE outcome_label IS NOT NULL
                    AND created_at > NOW() - INTERVAL '7 days'
                """)).scalar()
            if n >= 500:
                log.warning("retrain_threshold_reached", samples=n)
                # Publish to Kafka / trigger Airflow DAG in production
        except Exception:
            pass
