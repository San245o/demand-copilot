from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration.

    Every external dependency is optional. When a key is absent the corresponding
    integration falls back to a deterministic mock, so the app always runs.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # LLM
    google_api_key: str | None = None
    model_reasoning: str = "gemini-3.5-flash"
    model_narrative: str = "gemini-3.1-pro-preview"

    # Signal enrichers (all optional)
    tavily_api_key: str | None = None
    fred_api_key: str | None = None

    # CORS for the Next.js dev server
    frontend_origin: str = "http://localhost:3000"

    @property
    def has_llm(self) -> bool:
        return bool(self.google_api_key)

    @property
    def has_tavily(self) -> bool:
        return bool(self.tavily_api_key)

    @property
    def has_fred(self) -> bool:
        return bool(self.fred_api_key)


settings = Settings()
