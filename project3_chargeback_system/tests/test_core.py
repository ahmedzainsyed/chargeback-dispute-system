"""Tests for chargeback system."""
import pytest
import numpy as np

# ── Feature Engineering ────────────────────────────────────────────────────────
from src.features.engineer import single_vector, FEATURE_COLS

SAMPLE_TXN = {
    "transaction_id": "TXN0000000001",
    "merchant_id":    "MCH001",
    "card_bin":       "411111",
    "amount":         15000.0,
    "payment_method": "card",
    "bank":           "sbi",
    "country":        "IN",
    "device_type":    "android",
    "hour":           23,
    "day_of_week":    1,
    "is_weekend":     0,
    "is_chargeback":  0,
    "reason_category":"none",
}

def test_feature_vector_completeness():
    fv = single_vector(SAMPLE_TXN)
    for col in FEATURE_COLS:
        assert col in fv, f"Missing feature: {col}"
    assert all(isinstance(v, float) for v in fv.values())

def test_feature_risk_composite_range():
    fv = single_vector(SAMPLE_TXN)
    assert 0.0 <= fv["risk_composite"] <= 1.5

def test_feature_night_flag():
    txn = {**SAMPLE_TXN, "hour": 23}
    fv  = single_vector(txn)
    assert fv["is_night"] == 1.0

def test_feature_day_flag():
    txn = {**SAMPLE_TXN, "hour": 14}
    fv  = single_vector(txn)
    assert fv["is_night"] == 0.0

# ── Reason Classifier ──────────────────────────────────────────────────────────
from src.models.reason_classifier import ReasonClassifier

def test_reason_classifier_trains():
    clf = ReasonClassifier()
    result = clf.train()
    assert "classes" in result
    assert len(result["classes"]) == 5

def test_reason_classifier_fraud():
    clf = ReasonClassifier()
    clf.train()
    result = clf.classify("I never authorized this transaction")
    assert result["category"] == "fraud"   # category must be correct

def test_reason_classifier_not_received():
    clf = ReasonClassifier()
    clf.train()
    result = clf.classify("I never received the item I ordered")
    assert result["category"] == "not_received"

def test_reason_classifier_empty():
    clf = ReasonClassifier()
    clf.train()
    result = clf.classify("")
    assert result["category"] == "unrecognised"

# ── LLM Dispute Generator (template fallback) ─────────────────────────────────
from src.llm.dispute_generator import DisputeGenerator

def test_template_fallback():
    gen    = DisputeGenerator()   # no API key in test → template fallback
    txn    = {"transaction_id": "TXN001", "amount": 5000, "payment_method": "card",
              "device_type": "android"}
    cb     = {"chargeback_id": "CB001", "amount": 5000}
    result = gen._template("fraud", txn, cb)
    assert "CB001" in result
    assert len(result) > 100

def test_all_reason_templates():
    gen = DisputeGenerator()
    txn = {"transaction_id": "T1", "amount": 1000, "payment_method": "upi", "device_type": "ios"}
    cb  = {"chargeback_id": "CB1", "amount": 1000}
    for reason in ["fraud", "not_received", "duplicate", "product_issue", "unrecognised"]:
        result = gen._template(reason, txn, cb)
        assert isinstance(result, str) and len(result) > 50

# ── API Schemas ────────────────────────────────────────────────────────────────
from src.api.schemas import ChargebackRequest, OutcomeFeedback

def test_schema_valid():
    req = ChargebackRequest(
        transaction_id="TXN0000000001",
        chargeback_id="CB0001",
        amount=15000.0,
        merchant_id="MCH001",
    )
    assert req.amount == 15000.0

def test_schema_outcome():
    fb = OutcomeFeedback(chargeback_id="CB001", outcome=1)
    assert fb.outcome == 1
    assert fb.human_reviewed is False
