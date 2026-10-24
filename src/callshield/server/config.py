"""Server configuration and environment variable settings for Call Shield."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


_default_weights = Path("models/best.pt")
_has_trained_model = _default_weights.is_file()


class ServerSettings(BaseSettings):
    """Server environment settings."""

    CALLSHIELD_TOKEN: str = "secret-token"
    CALLSHIELD_DETECTOR: str = "torch" if _has_trained_model else "fake"
    MAX_UPLOAD_MB: int = 25
    MAX_CONCURRENT_INFERENCE: int = 4
    MAX_PENDING_WINDOWS: int = 2
    MODEL_WEIGHTS_PATH: Path | None = _default_weights if _has_trained_model else None
    DETECTOR_DELAY_MS: float = 0.0
    MAX_FRAME_BYTES: int = 65536
    MAX_SESSION_SECONDS: float = 3600.0
    MAX_SESSIONS: int = 100

    model_config = SettingsConfigDict(env_prefix="", case_sensitive=False)


settings = ServerSettings()
