class Settings:
    # Embedding server
    EMBEDDING_ENDPOINT: str = "http://127.0.0.1:8080/"

    # LLM - intent classifier
    INTENT_MODEL: str = "smollm2"
    INTENT_BASE_URL: str = "http://localhost:11434"
    INTENT_TEMPERATURE: float = 0
    INTENT_MAX_TOKENS: int = 126

    # LLM - schema mapper
    SCHEMA_MODEL: str = "qwen2.5-coder:3b"

    # Retriever
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    RETRIEVAL_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.7
    MAX_RETRIEVAL_ATTEMPTS: int = 1

    # Schema mapper embedding
    EMBED_MODEL_NAME: str = "all-MiniLM-L6-v2"
    TOP_K_CANDIDATES: int = 3
    TARGET_SCHEMA: list = [
        "user_id", "user_name", "phone_number", "email_address",
        "signup_date", "last_login", "is_active", "shipping_address",
    ]


settings = Settings()
