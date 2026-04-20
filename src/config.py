"""
Central configuration for ASIM-LLM-MAS.

Constants are plain module attributes. Access them as ``config.NAME``
so that runtime overrides (e.g. CLI --temperature) propagate to consumers.
"""
from dotenv import load_dotenv
load_dotenv()

import os

# ── Backtest ──────────────────────────────────────────────────────────────
INITIAL_CAPITAL: float = 1_000_000.0
MAX_NEW_CANDIDATES: int = 10
DEFAULT_MEMORY_CONTEXT: str = "FIRST PERIOD."

# ── LLM retries / temperatures ────────────────────────────────────────────
MAX_RETRIES: int = 2
RETRY_DELAY_SEC: float = 2.0
LLM_TEMPERATURE: float = 0.5          # screening / analysis; mutable via CLI
DECISION_TEMPERATURE: float = 0.1     # deterministic decisions

# ── LLM provider ──────────────────────────────────────────────────────────
MODEL_NAME: str = os.environ.get("KICONNECT_MODEL", "Mistral Small 3-2-24b Instruct KI:Inferenz.nrw")
API_KEY: str | None = os.environ.get("KICONNECT_API_KEY")
API_BASE_URL: str = "https://chat.kiconnect.nrw/api/v1"

# ── Fundamentals projection ───────────────────────────────────────────────
KEY_METRICS: list[str] = [
    # Valuation
    "pe_ratio", "pb_ratio", "market_cap", "enterprise_value",
    # Greenblatt
    "earnings_yield", "return_on_capital",
    # Profitability
    "roe", "roa", "net_income_margin", "revenue", "net_income", "operating_income",
    # Balance sheet
    "assets", "liabilities", "cash_equivalents", "current_ratio", "debt_to_equity",
    "current_assets", "current_liabilities", "long_term_debt",
    # Graham
    "ncav", "ncav_per_share", "book_value_per_share",
    # Growth
    "revenue_growth_yoy",
    # Price data
    "open", "high", "low", "close", "volume",
]


def snapshot() -> dict:
    """Return a JSON-serializable snapshot of all config values."""
    return {
        "INITIAL_CAPITAL": INITIAL_CAPITAL,
        "MAX_NEW_CANDIDATES": MAX_NEW_CANDIDATES,
        "DEFAULT_MEMORY_CONTEXT": DEFAULT_MEMORY_CONTEXT,
        "MAX_RETRIES": MAX_RETRIES,
        "RETRY_DELAY_SEC": RETRY_DELAY_SEC,
        "LLM_TEMPERATURE": LLM_TEMPERATURE,
        "DECISION_TEMPERATURE": DECISION_TEMPERATURE,
        "MODEL_NAME": MODEL_NAME,
        "API_BASE_URL": API_BASE_URL,
        "KEY_METRICS": KEY_METRICS,
    }
