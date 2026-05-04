"""TF-IDF + Logistic Regression reason classifier for dispute text."""
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import structlog

log = structlog.get_logger()

SAMPLES = {
    "fraud": [
        "I did not authorize this transaction",
        "Fraudulent transaction on my card",
        "Someone used my card without permission",
        "I never made this purchase",
        "Unauthorized payment from my account",
        "My card was used fraudulently",
        "This is a fraudulent charge",
        "I did not make this payment",
        "Somebody stole my card details",
        "Card was compromised and used without my consent",
    ],
    "not_received": [
        "I never received the item",
        "Product was not delivered",
        "Delivery not received",
        "Item not received after 30 days",
        "Package never arrived",
        "Order not delivered",
        "I have not received my order",
        "Service was not provided",
        "Goods never arrived at my address",
    ],
    "duplicate": [
        "Charged twice for same order",
        "Duplicate transaction",
        "Double debit from account",
        "Same amount deducted twice",
        "Duplicate charge on my statement",
        "Payment processed multiple times",
        "Charged multiple times for one order",
        "Two identical charges appeared",
    ],
    "product_issue": [
        "Product is defective",
        "Item received damaged",
        "Not as described on website",
        "Wrong item delivered",
        "Product quality issue",
        "Item is broken and unusable",
        "Received counterfeit product",
        "Significant difference from description",
    ],
    "unrecognised": [
        "I do not recognise this merchant",
        "Unknown charge on my statement",
        "I don't know what this payment is for",
        "Unrecognised transaction on account",
        "Cannot identify this charge",
        "Strange payment I didn't make",
    ],
}

class ReasonClassifier:
    def __init__(self):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2),
                                      sublinear_tf=True, min_df=1)),
            ("clf",   LogisticRegression(C=1.0, solver="lbfgs",
                                         max_iter=1000, class_weight="balanced"))
        ])

    def train(self) -> dict:
        os.makedirs("models", exist_ok=True)
        texts, labels = [], []
        for cat, samps in SAMPLES.items():
            texts.extend(samps); labels.extend([cat]*len(samps))

        X_tr, X_te, y_tr, y_te = train_test_split(
            texts, labels, test_size=0.2, stratify=labels, random_state=42
        )
        self.pipeline.fit(X_tr, y_tr)
        y_pred = self.pipeline.predict(X_te)
        log.info("reason_classifier_trained")
        print(classification_report(y_te, y_pred))
        joblib.dump(self, "models/reason_classifier.pkl")
        return {"classes": list(self.pipeline.classes_)}

    def classify(self, text: str) -> dict:
        if not text or not text.strip():
            return {"category": "unrecognised", "confidence": 0.5, "all_scores": {}}
        proba   = self.pipeline.predict_proba([text])[0]
        classes = self.pipeline.classes_
        top     = proba.argmax()
        return {
            "category":   classes[top],
            "confidence": round(float(proba[top]), 3),
            "all_scores": {c: round(float(p),3) for c,p in zip(classes, proba)},
        }

    @classmethod
    def load(cls, path="models/reason_classifier.pkl"):
        return joblib.load(path)
