import os

from dotenv import load_dotenv

# Load .env from project root (or above) if present
load_dotenv()  # does not override existing env vars by default


def get_env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None:
        raise RuntimeError(f"Required environment variable {name} is not set")
    return value
