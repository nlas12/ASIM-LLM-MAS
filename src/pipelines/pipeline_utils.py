"""
Pipeline Utilities — Shared Infrastructure
pipeline_utils.py
================================================================
Shared Pydantic models, state, memory, and LLM utilities
used by both single-agent and multi-agent pipelines.
Configuration constants live in :mod:`config`.
"""

import json
import re
from typing import Any, TypedDict
from dataclasses import dataclass, field

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

import config
from portfolio import TradeOrder
from experiment_logger import get_logger


# ══════════════════════════════════════════════════════════════════════════════
# Pydantic Schemas (LLM I/O)
# ══════════════════════════════════════════════════════════════════════════════

class ScreeningOutput(BaseModel):
    recheck_tickers: list[str] = Field(description="Current holdings to re-evaluate.")
    candidate_tickers: list[str] = Field(description="8-12 tickers for deep analysis.")
    rationale: str = Field(description="One sentence rationale.")

class CompanyAnalysis(BaseModel):
    ticker: str
    thesis: str = Field(description="One sentence thesis.")
    key_metrics: dict[str, Any]
    signal: str = Field(description="STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL")
    conviction: int = Field(ge=1, le=10)

class AnalysisOutput(BaseModel):
    analyses: list[CompanyAnalysis]

class ProposedTradeOrder(BaseModel):
    """LLM-proposed order (pre-execution).

    Allows HOLD with quantity=0; execution layer (portfolio.TradeOrder)
    enforces the stricter BUY/SELL-only, quantity>0 contract.
    """
    ticker: str
    action: str = Field(description="BUY, SELL, or HOLD")
    quantity: int = Field(ge=0, description="Number of shares. 0 for HOLD.")
    reasoning: str = Field(description="Max 15 words.")

class DecisionOutput(BaseModel):
    orders: list[ProposedTradeOrder]
    portfolio_rationale: str = Field(description="One sentence.")


# ══════════════════════════════════════════════════════════════════════════════
# Agent Memory
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TradeRecord:
    period: str; ticker: str; action: str; quantity: int; price: float; reasoning: str
    def to_dict(self):
        return {"period": self.period, "ticker": self.ticker, "action": self.action,
                "quantity": self.quantity, "price": self.price, "reasoning": self.reasoning}

@dataclass
class PositionMemory:
    ticker: str; entry_period: str; entry_price: float; current_shares: int
    cost_basis: float; unrealized_pnl: float; periods_held: int
    def to_dict(self):
        return {"ticker": self.ticker, "entry_period": self.entry_period,
                "entry_price": round(self.entry_price, 2), "current_shares": self.current_shares,
                "cost_basis": round(self.cost_basis, 2), "unrealized_pnl": round(self.unrealized_pnl, 2),
                "periods_held": self.periods_held}

@dataclass
class AgentMemory:
    trade_history: list[TradeRecord] = field(default_factory=list)
    position_tracker: dict[str, PositionMemory] = field(default_factory=dict)
    period_returns: list[dict] = field(default_factory=list)
    past_rationales: list[dict] = field(default_factory=list)

    def record_trade(self, period, order: TradeOrder, exec_price: float):
        self.trade_history.append(TradeRecord(
            period=period, ticker=order.ticker, action=order.action,
            quantity=order.quantity, price=exec_price, reasoning=order.reasoning))
        if order.action == "BUY":
            if order.ticker in self.position_tracker:
                pos = self.position_tracker[order.ticker]
                if pos.current_shares <= 0:
                    # Stale entry (fully sold but not cleaned up) — reset as new position
                    self.position_tracker[order.ticker] = PositionMemory(
                        ticker=order.ticker, entry_period=period, entry_price=exec_price,
                        current_shares=order.quantity, cost_basis=exec_price * order.quantity,
                        unrealized_pnl=0.0, periods_held=0)
                else:
                    pos.cost_basis += exec_price * order.quantity
                    pos.current_shares += order.quantity
                    pos.entry_price = pos.cost_basis / pos.current_shares
            else:
                self.position_tracker[order.ticker] = PositionMemory(
                    ticker=order.ticker, entry_period=period, entry_price=exec_price,
                    current_shares=order.quantity, cost_basis=exec_price * order.quantity,
                    unrealized_pnl=0.0, periods_held=0)
        elif order.action == "SELL":
            if order.ticker in self.position_tracker:
                pos = self.position_tracker[order.ticker]
                if pos.current_shares > 0:
                    sell_fraction = min(order.quantity / pos.current_shares, 1.0)
                    pos.cost_basis *= (1 - sell_fraction)
                pos.current_shares -= order.quantity
                if pos.current_shares <= 0: del self.position_tracker[order.ticker]

    def update_period_end(self, period, portfolio_value, price_data, initial_capital):
        for ticker, pos in self.position_tracker.items():
            current_price = price_data.get(ticker, pos.entry_price)
            pos.unrealized_pnl = (current_price - pos.entry_price) * pos.current_shares
            pos.periods_held += 1
        prev_value = self.period_returns[-1]["value"] if self.period_returns else initial_capital
        ret_pct = ((portfolio_value - prev_value) / prev_value) * 100
        self.period_returns.append({"period": period, "value": round(portfolio_value, 2), "return_pct": round(ret_pct, 2)})

    def record_rationale(self, period, rationale):
        self.past_rationales.append({"period": period, "rationale": rationale})

    def to_prompt_context(self):
        """Build memory context string for LLM prompts."""
        context_parts = []

        # Current positions
        if self.position_tracker:
            position_lines = ["POSITIONS:"]
            for pos in self.position_tracker.values():
                pnl_pct = (pos.unrealized_pnl / pos.cost_basis * 100) if pos.cost_basis > 0 else 0
                position_lines.append(
                    f"  {pos.ticker}: {pos.current_shares}sh, entry=${pos.entry_price:.0f}, "
                    f"P&L={pnl_pct:+.1f}%, held {pos.periods_held}p"
                )
            context_parts.append("\n".join(position_lines))

        # Recent trades
        if self.trade_history:
            trade_lines = ["TRADES:"]
            for trade in self.trade_history[-6:]:
                trade_lines.append(
                    f"  {trade.period}: {trade.action} {trade.ticker} x{trade.quantity} @${trade.price:.0f}"
                )
            context_parts.append("\n".join(trade_lines))

        # Recent returns
        if self.period_returns:
            returns_str = ", ".join(
                f"{p['period']}:{p['return_pct']:+.1f}%" for p in self.period_returns[-4:]
            )
            context_parts.append(f"RETURNS: {returns_str}")

        return "\n".join(context_parts) if context_parts else config.DEFAULT_MEMORY_CONTEXT


# ══════════════════════════════════════════════════════════════════════════════
# LangGraph State
# ══════════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict, total=False):
    portfolio: dict
    cash: float
    market_universe: list
    period_label: str
    persona_prompt: str
    memory_context: str
    screening: dict
    analyses: dict
    decision: dict
    error: str
    fundamentals_db: dict
    price_data: dict
    step_status: dict  # Tracks step skips/failures: {"analysis": "skipped: no screening", ...}


# ══════════════════════════════════════════════════════════════════════════════
# LLM Factory + JSON Parsing
# ══════════════════════════════════════════════════════════════════════════════

def make_llm(temperature: float | None = None) -> ChatOpenAI:
    effective_temp = temperature if temperature is not None else config.LLM_TEMPERATURE
    return ChatOpenAI(
        model=config.MODEL_NAME,
        temperature=effective_temp,
        openai_api_key=config.API_KEY,
        openai_api_base=config.API_BASE_URL,
    )

def parse_llm_json(raw_text):
    try: return json.loads(raw_text.strip())
    except json.JSONDecodeError: pass
    cleaned = re.sub(r"```(?:json)?\s*", "", raw_text).strip().rstrip("`").strip()
    try: return json.loads(cleaned)
    except json.JSONDecodeError: pass
    for pattern in [r'(\{[\s\S]*\})', r'(\[[\s\S]*\])']:
        match = re.search(pattern, raw_text)
        if match:
            try: return json.loads(match.group(1))
            except json.JSONDecodeError: pass
    repaired = re.sub(r',\s*([}\]])', r'\1', cleaned).replace("'", '"')
    try: return json.loads(repaired)
    except json.JSONDecodeError: pass
    raise ValueError(f"Could not parse JSON from LLM output:\n{raw_text[:500]}")

def _log(step, sys_prompt, human_prompt, raw_output, parsed, success, error="", temperature=None, logger=None):
    if logger is None:
        logger = get_logger()
    if logger:
        logged_temp = temperature if temperature is not None else config.LLM_TEMPERATURE
        logger.log_llm_call(step=step, system_prompt=sys_prompt, human_prompt=human_prompt,
                            raw_output=raw_output or "", parsed_output=parsed,
                            success=success, error=error, temperature=logged_temp)
