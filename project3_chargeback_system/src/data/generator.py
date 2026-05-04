"""
Realistic synthetic chargeback dataset.
Run: python -m src.data.generator --n 50000
"""
import random, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

random.seed(42); np.random.seed(42)

REASON_DIST = {
    "fraud":         0.45,
    "not_received":  0.20,
    "duplicate":     0.15,
    "product_issue": 0.12,
    "unrecognised":  0.08,
}
REASON_CODES = {
    "fraud":         ["4853", "10.4", "10.5"],
    "not_received":  ["13.1", "13.2"],
    "duplicate":     ["12.6", "12.7"],
    "product_issue": ["13.3", "13.5"],
    "unrecognised":  ["10.1", "10.2"],
}
BANKS   = ["hdfc", "icici", "sbi", "axis", "kotak", "other"]
METHODS = ["upi", "card", "netbanking"]
DEVICES = ["android", "ios", "web"]

def cb_prob(amount: float, method: str, hour: int, bank: str) -> float:
    p = 0.015
    if amount > 50_000: p *= 3.0
    elif amount > 10_000: p *= 1.8
    elif amount > 5_000:  p *= 1.3
    if method == "card":       p *= 1.5
    elif method == "netbanking": p *= 0.8
    if 23 <= hour or hour <= 4: p *= 2.0
    elif 5 <= hour <= 9:        p *= 1.3
    if bank == "sbi":           p *= 1.4
    elif bank in ("hdfc","icici"): p *= 0.8
    return min(p, 0.25)

def generate(n: int = 50_000) -> pd.DataFrame:
    cutoff  = datetime.now() - timedelta(days=365)
    records = []
    for i in range(n):
        t      = cutoff + timedelta(seconds=random.randint(0, 365*86400))
        amount = float(np.clip(np.random.lognormal(7.5, 1.2), 100, 500_000))
        method = random.choices(METHODS, weights=[0.50, 0.35, 0.15])[0]
        bank   = random.choices(BANKS,   weights=[0.28, 0.22, 0.20, 0.15, 0.10, 0.05])[0]
        hour   = t.hour
        is_cb  = int(random.random() < cb_prob(amount, method, hour, bank))
        cat    = random.choices(list(REASON_DIST), weights=list(REASON_DIST.values()))[0] if is_cb else "none"
        code   = random.choice(REASON_CODES.get(cat, [""])) if is_cb else ""

        records.append({
            "transaction_id": f"TXN{i:010d}",
            "merchant_id":    f"MCH{random.randint(1,200):04d}",
            "card_bin":       str(random.randint(400000, 599999)),
            "amount":         round(amount, 2),
            "payment_method": method,
            "bank":           bank,
            "country":        random.choices(["IN","US","GB","AE"], weights=[0.88,0.05,0.04,0.03])[0],
            "device_type":    random.choice(DEVICES),
            "hour":           hour,
            "day_of_week":    t.weekday(),
            "is_weekend":     int(t.weekday() >= 5),
            "is_chargeback":  is_cb,
            "reason_category":cat,
            "reason_code":    code,
        })

    df = pd.DataFrame(records)
    print(f"Generated {len(df):,} | chargeback rate: {df['is_chargeback'].mean():.3%}")
    return df

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50_000)
    ap.add_argument("--out", default="data/chargeback_data.csv")
    args = ap.parse_args()
    Path("data").mkdir(exist_ok=True)
    df = generate(args.n)
    df.to_csv(args.out, index=False)
    print(f"Saved → {args.out}")
