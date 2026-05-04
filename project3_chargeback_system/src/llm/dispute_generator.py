"""
LLM-powered dispute response generator.
Uses Anthropic Claude API with RAG context.
Falls back to deterministic templates if no API key.
"""
import structlog
from typing import Dict, Any, List
from config.settings import settings

log = structlog.get_logger()

SYSTEM = """You are a payments dispute specialist writing chargeback responses.
Your responses must be:
- Professional and factual (under 400 words)
- Based ONLY on evidence provided — never invent facts
- Structured: Reference → Evidence → Argument → Request for reversal
Write in formal business English."""

GUIDELINES = {
    "fraud":         "Highlight: 3DS/OTP authentication, device fingerprint match, customer IP, delivery timestamp.",
    "not_received":  "Highlight: tracking number, delivery confirmation, GPS/courier proof, signed acknowledgement.",
    "duplicate":     "Highlight: unique transaction IDs, system logs showing single debit, different timestamps.",
    "product_issue": "Highlight: product description match, quality records, return policy communicated, merchant-customer communications.",
    "unrecognised":  "Highlight: authentication logs, customer session, device used, IP address consistency.",
}

TEMPLATES = {
    "fraud":
        "Dear Chargeback Department,\n\nWe formally dispute chargeback {cb_id} for transaction {txn_id} "
        "amounting to ₹{amount}.\n\nThe transaction was processed on {method} with full authentication "
        "including OTP verification and 3DS compliance. Our system logs confirm the transaction was "
        "initiated from a registered device (ID: {device}) with consistent IP geolocation matching the "
        "cardholder's registered address.\n\nWe respectfully request reversal of this chargeback and "
        "enclose all authentication records for your review.\n\nYours faithfully,\nMerchant Risk Team",
    "not_received":
        "Dear Chargeback Department,\n\nWe dispute chargeback {cb_id}. Order delivery has been confirmed "
        "per our logistics records. Tracking details and delivery confirmation are available upon request. "
        "The shipment was dispatched within the agreed SLA.\n\nWe request reversal of this chargeback.\n\n"
        "Yours faithfully,\nMerchant Risk Team",
    "duplicate":
        "Dear Chargeback Department,\n\nWe dispute chargeback {cb_id}. Our payment gateway logs confirm "
        "only one successful debit for transaction {txn_id}. Each transaction carries a unique reference ID. "
        "System records are available for audit.\n\nWe request reversal.\n\nYours faithfully,\nMerchant Risk Team",
    "product_issue":
        "Dear Chargeback Department,\n\nWe dispute chargeback {cb_id}. The product/service was delivered "
        "as described. Our quality control and delivery records confirm fulfilment per agreement. "
        "We have not received a valid return request.\n\nWe request reversal.\n\nYours faithfully,\nMerchant Risk Team",
    "unrecognised":
        "Dear Chargeback Department,\n\nWe dispute chargeback {cb_id}. Transaction {txn_id} was authenticated "
        "via {method} with OTP and session verification. Device and IP records match cardholder's profile. "
        "We request reversal.\n\nYours faithfully,\nMerchant Risk Team",
}

class DisputeGenerator:
    def __init__(self):
        self._client = None
        if settings.USE_LLM:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
                log.info("llm_client_ready")
            except Exception as e:
                log.warning("llm_init_failed", error=str(e))

    def generate(self, txn: Dict[str, Any], cb: Dict[str, Any],
                 similar_cases: List[Dict], reason: str) -> str:
        if self._client:
            return self._llm_generate(txn, cb, similar_cases, reason)
        return self._template(reason, txn, cb)

    def _llm_generate(self, txn, cb, similar_cases, reason) -> str:
        evidence = "\n".join(f"- {k}: {v}" for k,v in txn.items() if v)
        cases_txt = "\n\n".join(
            f"[{c.get('outcome','?')}]: {str(c.get('dispute_response',''))[:200]}..."
            for c in similar_cases[:2]
        ) or "No similar cases."
        guidance = GUIDELINES.get(reason, "Provide all relevant evidence.")

        prompt = (
            f"Generate a chargeback dispute letter.\n\n"
            f"TRANSACTION EVIDENCE:\n{evidence}\n\n"
            f"CHARGEBACK:\n- ID: {cb.get('chargeback_id')}\n"
            f"- Amount: ₹{cb.get('amount')}\n- Reason: {reason}\n"
            f"- Deadline: {cb.get('deadline_at')}\n\n"
            f"GUIDANCE: {guidance}\n\n"
            f"SIMILAR CASES (structure reference only):\n{cases_txt}\n\n"
            f"Write the dispute letter now."
        )
        try:
            resp = self._client.messages.create(
                model=settings.LLM_MODEL,
                max_tokens=settings.LLM_MAX_TOKENS,
                system=SYSTEM,
                messages=[{"role": "user", "content": prompt}]
            )
            return resp.content[0].text
        except Exception as e:
            log.error("llm_call_failed", error=str(e))
            return self._template(reason, txn, cb)

    def _template(self, reason: str, txn: Dict, cb: Dict) -> str:
        tmpl = TEMPLATES.get(reason, TEMPLATES["fraud"])
        return tmpl.format(
            cb_id=cb.get("chargeback_id", "N/A"),
            txn_id=txn.get("transaction_id", "N/A"),
            amount=txn.get("amount", "N/A"),
            method=txn.get("payment_method", "our payment system"),
            device=txn.get("device_type", "registered device"),
        )
