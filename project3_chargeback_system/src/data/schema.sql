-- Chargeback System Database Schema
-- Run: psql -U razorpay -d chargebacks -f src/data/schema.sql

CREATE TABLE IF NOT EXISTS transactions (
    id              SERIAL PRIMARY KEY,
    transaction_id  VARCHAR(64) UNIQUE NOT NULL,
    merchant_id     VARCHAR(32) NOT NULL,
    card_bin        VARCHAR(6),
    amount          DECIMAL(12,2),
    currency        CHAR(3) DEFAULT 'INR',
    payment_method  VARCHAR(20),
    bank            VARCHAR(20),
    country         CHAR(2) DEFAULT 'IN',
    device_type     VARCHAR(20),
    hour            SMALLINT,
    day_of_week     SMALLINT,
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chargebacks (
    id                  SERIAL PRIMARY KEY,
    transaction_id      VARCHAR(64) REFERENCES transactions(transaction_id) ON DELETE SET NULL,
    chargeback_id       VARCHAR(64) UNIQUE NOT NULL,
    reason_code         VARCHAR(10),
    reason_category     VARCHAR(30),
    amount              DECIMAL(12,2),
    filed_at            TIMESTAMP DEFAULT NOW(),
    deadline_at         TIMESTAMP,
    status              VARCHAR(20) DEFAULT 'open',
    risk_score          FLOAT,
    risk_tier           VARCHAR(10),
    dispute_response    TEXT,
    human_reviewed      BOOLEAN DEFAULT FALSE,
    outcome_label       SMALLINT,
    created_at          TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS dispute_evidence (
    id              SERIAL PRIMARY KEY,
    chargeback_id   VARCHAR(64) REFERENCES chargebacks(chargeback_id) ON DELETE CASCADE,
    evidence_type   VARCHAR(30),
    evidence_text   TEXT,
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cb_txn      ON chargebacks(transaction_id);
CREATE INDEX IF NOT EXISTS idx_cb_status   ON chargebacks(status);
CREATE INDEX IF NOT EXISTS idx_cb_tier     ON chargebacks(risk_tier);
CREATE INDEX IF NOT EXISTS idx_cb_filed    ON chargebacks(filed_at);
CREATE INDEX IF NOT EXISTS idx_cb_outcome  ON chargebacks(outcome_label);
