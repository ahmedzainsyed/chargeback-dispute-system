"""
Train all models for the chargeback system in one shot.
Run: python train_all.py
"""
import os
from pathlib import Path

def main():
    Path("data").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    print("=" * 55)
    print("  STEP 1: Generating synthetic data (50,000 txns)")
    print("=" * 55)
    from src.data.generator import generate
    df = generate(50_000)
    df.to_csv("data/chargeback_data.csv", index=False)

    print("\n" + "=" * 55)
    print("  STEP 2: Training XGBoost chargeback predictor")
    print("=" * 55)
    from src.models.chargeback_model import ChargebackPredictor
    predictor = ChargebackPredictor()
    metrics   = predictor.train(df)
    print(f"  AUC-PR: {metrics['auc_pr']:.4f} | AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"  Optimal threshold: {metrics['threshold']:.4f}")

    print("\n" + "=" * 55)
    print("  STEP 3: Training NLP reason classifier")
    print("=" * 55)
    from src.models.reason_classifier import ReasonClassifier
    clf = ReasonClassifier()
    clf.train()

    print("\n" + "=" * 55)
    print("  ALL MODELS TRAINED ✓")
    print("  Run: uvicorn src.api.app:app --port 8002")
    print("=" * 55)

if __name__ == "__main__":
    main()
