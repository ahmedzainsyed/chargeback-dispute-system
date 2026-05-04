"""18-feature engineering for chargeback prediction."""
import numpy as np
import pandas as pd
from typing import Dict, Any

FEATURE_COLS = [
    "amount_log", "is_high_amount", "is_international", "card_bin_risk",
    "hour_sin", "hour_cos", "is_weekend", "is_night",
    "is_card", "is_upi", "is_netbanking",
    "bank_risk_score",
    "merchant_chargeback_rate", "merchant_txn_count_log",
    "amount_vs_merchant_avg", "is_first_time_card",
    "is_mobile",
    "risk_composite",
]

BANK_RISK = {
    "sbi": 0.70, "other": 0.60, "axis": 0.40,
    "kotak": 0.30, "icici": 0.25, "hdfc": 0.20,
}

def _merchant_stats(df: pd.DataFrame) -> pd.DataFrame:
    stats = (
        df.groupby("merchant_id")
        .agg(
            merchant_chargeback_rate=("is_chargeback", "mean"),
            merchant_txn_count=("transaction_id", "count"),
            merchant_avg_amount=("amount", "mean"),
        )
        .reset_index()
    )
    return df.merge(stats, on="merchant_id", how="left")

def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "merchant_chargeback_rate" not in df.columns:
        df = _merchant_stats(df)

    df["amount_log"]    = np.log1p(df["amount"]) / np.log1p(500_000)
    df["is_high_amount"]= (df["amount"] > 10_000).astype(int)
    df["is_international"] = (df["country"] != "IN").astype(int)
    df["card_bin_risk"] = df["card_bin"].astype(str).str[:1].apply(
        lambda x: 0.8 if x in ("4","5") else 0.4
    )

    hour = df["hour"].astype(float)
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["is_night"] = ((hour >= 22) | (hour <= 5)).astype(int)

    df["is_card"]      = (df["payment_method"] == "card").astype(int)
    df["is_upi"]       = (df["payment_method"] == "upi").astype(int)
    df["is_netbanking"]= (df["payment_method"] == "netbanking").astype(int)

    df["bank_risk_score"] = df["bank"].map(BANK_RISK).fillna(0.5)

    df["merchant_chargeback_rate"] = df.get("merchant_chargeback_rate", 0.015).fillna(0.015) \
        if hasattr(df.get("merchant_chargeback_rate", 0.015), "fillna") \
        else df["merchant_chargeback_rate"].fillna(0.015)
    df["merchant_txn_count_log"]   = np.log1p(df.get("merchant_txn_count", 1).fillna(1) \
        if hasattr(df.get("merchant_txn_count",1),"fillna") else df["merchant_txn_count"].fillna(1))

    avg = df.get("merchant_avg_amount", 2000)
    avg = avg.fillna(2000) if hasattr(avg, "fillna") else df["merchant_avg_amount"].fillna(2000)
    df["amount_vs_merchant_avg"] = df["amount"] / (avg + 1)

    df["is_mobile"]         = df["device_type"].isin(["android","ios"]).astype(int)
    df["is_first_time_card"]= (df["card_bin"].astype(str) == "000000").astype(int)

    df["risk_composite"] = (
        df["amount_log"]           * 0.30 +
        df["bank_risk_score"]      * 0.20 +
        df["is_night"]             * 0.20 +
        df["is_card"]              * 0.15 +
        df["is_international"]     * 0.15
    )
    return df

def single_vector(txn: Dict[str, Any]) -> Dict[str, float]:
    df = pd.DataFrame([txn])
    for col in ["merchant_chargeback_rate", "merchant_txn_count", "merchant_avg_amount",
                "is_chargeback", "is_weekend", "day_of_week"]:
        if col not in df.columns:
            df[col] = 0
    df = engineer(df)
    return {c: float(df[c].iloc[0]) for c in FEATURE_COLS}
