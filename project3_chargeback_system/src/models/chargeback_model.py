"""Cost-sensitive XGBoost chargeback predictor."""
import os
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix
)
from sklearn.calibration import CalibratedClassifierCV
from src.features.engineer import engineer, FEATURE_COLS
import structlog

log = structlog.get_logger()

class ChargebackPredictor:
    def __init__(self):
        self.model     = None
        self.threshold = 0.45

    def _sample_weights(self, y, amounts):
        base = np.where(y == 1, 4.0, 1.0)
        amt  = np.log1p(amounts) / np.log1p(500_000)
        return base * (1 + amt)

    def _cost_threshold(self, y_true, y_proba) -> float:
        best, best_t = float("inf"), 0.5
        for t in np.arange(0.1, 0.9, 0.01):
            pred = (y_proba >= t).astype(int)
            fp   = ((pred==1)&(y_true==0)).sum()
            fn   = ((pred==0)&(y_true==1)).sum()
            cost = 1.0*fp + 4.0*fn
            if cost < best:
                best, best_t = cost, t
        return best_t

    def train(self, df: pd.DataFrame) -> dict:
        os.makedirs("models", exist_ok=True)
        df = engineer(df)
        X, y = df[FEATURE_COLS].fillna(0), df["is_chargeback"].astype(int)
        X_tr, X_te, y_tr, y_te, a_tr, _ = train_test_split(
            X, y, df["amount"], test_size=0.2, stratify=y, random_state=42
        )
        sw = self._sample_weights(y_tr.values, a_tr.values)
        neg, pos = (y_tr==0).sum(), (y_tr==1).sum()
        log.info("training", neg=int(neg), pos=int(pos))

        base = xgb.XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="aucpr", early_stopping_rounds=25,
            random_state=42,
        )
        base.fit(X_tr, y_tr, sample_weight=sw, eval_set=[(X_te, y_te)], verbose=50)

        cal = CalibratedClassifierCV(base, cv="prefit", method="sigmoid")
        cal.fit(X_te, y_te)
        self.model = cal

        y_prob = cal.predict_proba(X_te)[:, 1]
        self.threshold = self._cost_threshold(y_te.values, y_prob)
        auc_pr = average_precision_score(y_te, y_prob)
        auc_roc= roc_auc_score(y_te, y_prob)

        log.info("trained", auc_pr=round(auc_pr,4), auc_roc=round(auc_roc,4),
                 threshold=round(self.threshold,4))
        print(classification_report(y_te, (y_prob>=self.threshold).astype(int)))

        joblib.dump(self, "models/chargeback_model.pkl")
        return {"auc_pr": auc_pr, "auc_roc": auc_roc, "threshold": self.threshold}

    def predict(self, fv: dict) -> dict:
        X = pd.DataFrame([fv])[FEATURE_COLS].fillna(0)
        p = float(self.model.predict_proba(X)[0][1])
        tier = "HIGH" if p >= 0.75 else ("MEDIUM" if p >= 0.45 else "LOW")
        return {"score": round(p,4), "tier": tier, "flag": p >= self.threshold}

    @classmethod
    def load(cls, path="models/chargeback_model.pkl"):
        return joblib.load(path)
