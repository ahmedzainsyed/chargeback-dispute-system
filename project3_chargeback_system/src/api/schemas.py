from pydantic import BaseModel, Field
from typing import Optional

class ChargebackRequest(BaseModel):
    transaction_id:         str
    chargeback_id:          str
    amount:                 float = Field(..., gt=0)
    payment_method:         str   = "card"
    bank:                   str   = "hdfc"
    country:                str   = "IN"
    device_type:            str   = "android"
    hour:                   int   = Field(default=12, ge=0, le=23)
    merchant_id:            str   = "MCH000"
    is_weekend:             int   = 0
    customer_dispute_text:  Optional[str] = None
    reason_category:        Optional[str] = None

class ChargebackResponse(BaseModel):
    chargeback_id:          str
    transaction_id:         str
    risk_score:             float
    risk_tier:              str
    reason_category:        str
    reason_confidence:      float
    human_review_required:  bool
    dispute_response:       Optional[str] = None

class OutcomeFeedback(BaseModel):
    chargeback_id:  str
    outcome:        int   # 1 = merchant won, 0 = bank won
    human_reviewed: bool  = False
