"""Environment-driven settings. Read lazily so tests can override os.environ."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
STATIC_DIR = ROOT / "static"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# LLM call limits
LLM_TIMEOUT_S = 20
LLM_MAX_RETRIES = 1
LLM_MAX_TOKENS = 700
HISTORY_WINDOW = 20
SLOW_TURN_MS = 10_000
LARGE_CHANGE_PCT = 30.0


def openrouter_api_key() -> str:
    return os.getenv("OPENROUTER_API_KEY", "")


def groq_api_key() -> str:
    return os.getenv("GROQ_API_KEY", "")


def parse_model_spec(spec: str) -> tuple[str, str]:
    """'groq:llama-3.3-70b-versatile' -> ('groq', 'llama-3.3-70b-versatile'); anything else is an OpenRouter model ID
    (which may itself contain ':' like 'vendor/model:free')."""
    for provider in ("groq", "openrouter"):
        if spec.lower().startswith(provider + ":"):
            return provider, spec[len(provider) + 1:]
    return "openrouter", spec


def provider_api_key(provider: str) -> str:
    return groq_api_key() if provider == "groq" else openrouter_api_key()


def primary_model() -> str:
    return os.getenv("PRIMARY_MODEL", "poolside/laguna-xs-2.1:free")


def fallback_models() -> list[str]:
    """FALLBACK_MODEL may be a comma-separated chain, tried in order after the primary."""
    raw = os.getenv("FALLBACK_MODEL", "cohere/north-mini-code:free,inclusionai/ling-3.0-flash-fin:free,openrouter/free")
    return [m.strip() for m in raw.split(",") if m.strip()]


def admin_password() -> str:
    return os.getenv("ADMIN_PASSWORD", "")


def db_path() -> str:
    p = os.getenv("DB_PATH", "app.db")
    return p if os.path.isabs(p) else str(ROOT / p)


def configured_models() -> list[str]:
    """Primary followed by the fallback chain, exactly as configured."""
    return [primary_model(), *fallback_models()]


def model_chain() -> list[str]:
    """Configured models whose provider has an API key. The first entry is the effective primary."""
    return [m for m in configured_models() if provider_api_key(parse_model_spec(m)[0])]


def skipped_models() -> list[str]:
    return [m for m in configured_models() if m not in model_chain()]
