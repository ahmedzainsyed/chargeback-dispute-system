from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql://razorpay:razorpay@localhost:5432/chargebacks"
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379

    ANTHROPIC_API_KEY: str = ""
    LLM_MODEL: str = "claude-sonnet-4-20250514"
    LLM_MAX_TOKENS: int = 800

    CHARGEBACK_HIGH_RISK_THRESHOLD: float   = 0.75
    CHARGEBACK_MEDIUM_RISK_THRESHOLD: float = 0.45
    COST_WEIGHT_FP: float = 1.0
    COST_WEIGHT_FN: float = 4.0

    RAG_TOP_K: int = 3
    API_SECRET_KEY: str = "change-me-in-prod"
    HUMAN_REVIEW_QUEUE_KEY: str = "chargeback:human_review"

    # Fallback: use templates if no Anthropic key set
    USE_LLM: bool = True

    class Config:
        env_file = ".env"

    def model_post_init(self, __context):
        if not self.ANTHROPIC_API_KEY:
            object.__setattr__(self, "USE_LLM", False)

settings = Settings()
