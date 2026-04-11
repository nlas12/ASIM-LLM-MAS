"""
Single-Agent Investment Pipeline
single_agent_pipeline.py
================================================================
"""
from dotenv import load_dotenv
load_dotenv()

import json
import re
import os
import time
import copy
from typing import Any, Optional, TypedDict, Annotated
from dataclasses import dataclass, field

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from portfolio import Portfolio
from experiment_logger import get_logger
from personas import PERSONAS


# ══════════════════════════════════════════════════════════════════════════════
# Pydantic Schemas
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

class TradeOrder(BaseModel):
    ticker: str
    action: str = Field(description="BUY, SELL, or HOLD")
    quantity: int = Field(ge=0, description="Number of shares. 0 for HOLD.")
    reasoning: str = Field(description="Max 15 words.")

class DecisionOutput(BaseModel):
    orders: list[TradeOrder]
    portfolio_rationale: str = Field(description="One sentence.")


# ══════════════════════════════════════════════════════════════════════════════
# Retry / Fallback
# ══════════════════════════════════════════════════════════════════════════════

def _safe_screening_fallback(state: dict) -> dict:
    holdings = state.get("portfolio", {})
    recheck = list(holdings.keys())
    universe = state.get("market_universe", [])
    universe_tickers = [s["ticker"] for s in universe if "ticker" in s]
    candidates = [t for t in universe_tickers if t not in recheck][:10]
    fallback = ScreeningOutput(recheck_tickers=recheck, candidate_tickers=candidates,
                               rationale="Fallback: LLM screening failed after retry.")
    return {"screening": fallback.model_dump()}

def _safe_analysis_fallback() -> dict:
    return {"analyses": {"analyses": []}}

def _safe_decision_fallback() -> dict:
    return {"decision": {"orders": [], "portfolio_rationale": "Fallback: LLM decision failed after retry. Holding."}}


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
        
        return "\n".join(context_parts) if context_parts else DEFAULT_MEMORY_CONTEXT


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
# Configuration Constants
# ══════════════════════════════════════════════════════════════════════════════

MAX_RETRIES = 2
RETRY_DELAY_SEC = 2.0

LLM_TEMPERATURE: float = 0.5
DECISION_TEMPERATURE: float = 0.1
DEFAULT_MEMORY_CONTEXT: str = "FIRST PERIOD."

# Prompt Templates
SCREENING_SYSTEM_TEMPLATE = """{persona_prompt}

Step 1: SCREENING for {period_label}.
{memory_context}

RULES:
- recheck_tickers: current holdings to re-evaluate.
- candidate_tickers: 8-12 best candidates per your philosophy.
- Output ONLY JSON:
{{"recheck_tickers":["TICK_A1"],"candidate_tickers":["TICK_B1","TICK_B2"],"rationale":"One sentence."}}"""

ANALYSIS_SYSTEM_TEMPLATE = """{persona_prompt}

Step 2: ANALYSIS for {period_label}.
{memory_context}

For each candidate output:
- ticker, thesis (1 sentence max 20 words), key_metrics (3-5 numbers), signal (STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL), conviction (1-10)
Output ONLY JSON:
{{"analyses":[{{"ticker":"TICK_XX","thesis":"...","key_metrics":{{"pe":18.5,"roe":0.25}},"signal":"BUY","conviction":7}}]}}"""

DECISION_SYSTEM_TEMPLATE = """{persona_prompt}

Step 3: TRADE DECISIONS for {period_label}.
{memory_context}

RULES:
1. For each analysed ticker decide: BUY, SELL, or HOLD.
2. HOLD means no action on that ticker — quantity must be 0.
3. It is valid to HOLD all positions if no trade improves the portfolio.
4. Each order: ticker, action (BUY/SELL/HOLD), quantity (int>=0, 0 for HOLD), reasoning (max 15 words).
5. BUY total cost <= available cash. SELL quantity <= held shares.
6. Size by conviction: highest conviction = largest position.
7. Output ONLY JSON:
{{"orders":[{{"ticker":"TICK_XX","action":"BUY","quantity":100,"reasoning":"..."}}],"portfolio_rationale":"One sentence."}}"""

# Human Prompt Templates
SCREENING_HUMAN_TEMPLATE = """Portfolio: {portfolio_json}
Cash: ${cash:,.0f}

UNIVERSE:
{universe_str}"""

ANALYSIS_HUMAN_TEMPLATE = """Candidates: {candidates}
Portfolio: {portfolio_json}
Cash: ${cash:,.0f}

DATA:
{fundamentals_str}"""

DECISION_HUMAN_TEMPLATE = """Cash: ${cash:,.0f}
Holdings: {portfolio_json}

ANALYSIS:
{analysis_str}

Produce trade orders (BUY/SELL/HOLD) as JSON now."""

# ══════════════════════════════════════════════════════════════════════════════
# LLM Factory + JSON Parsing
# ══════════════════════════════════════════════════════════════════════════════

def make_llm(temperature=None):
    effective_temp = temperature if temperature is not None else LLM_TEMPERATURE
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=effective_temp,
                                   google_api_key=os.environ.get("GOOGLE_API_KEY"),
                                   convert_system_message_to_human=False)

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


# ══════════════════════════════════════════════════════════════════════════════
# Node Implementations
# ══════════════════════════════════════════════════════════════════════════════

_KEY_METRICS = [
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

def _format_universe(market_universe):
    if not market_universe: return "[]"
    return json.dumps(market_universe, indent=1)

def _log(step, sys_prompt, human_prompt, raw_output, parsed, success, error="", temperature=None):
    logger = get_logger()
    if logger:
        logged_temp = temperature if temperature is not None else LLM_TEMPERATURE
        logger.log_llm_call(step=step, system_prompt=sys_prompt, human_prompt=human_prompt,
                            raw_output=raw_output or "", parsed_output=parsed,
                            success=success, error=error, temperature=logged_temp)
        
def _invoke_llm_with_retries(step_name: str, system_content: str, human_content: str,
                              output_model: type, fallback_func: callable,
                              temperature: Optional[float] = None) -> dict:
    """
    Generic LLM invocation with retry logic, JSON parsing, validation, and logging.
    Reduces code duplication across screening, analysis, and decision nodes.
    """
    system = SystemMessage(content=system_content)
    human = HumanMessage(content=human_content)
    raw_content = ""
    
    for attempt in range(MAX_RETRIES):
        try:
            llm = make_llm(temperature=temperature)
            response = llm.invoke([system, human])
            raw_content = response.content
            data = parse_llm_json(raw_content)
            result = output_model(**data)
            _log(step_name, system_content, human_content, raw_content, data, True, temperature=temperature)
            return {step_name: result.model_dump()}
        except Exception as e:
            _log(step_name, system_content, human_content, raw_content,
                 None, False, error=f"Attempt {attempt+1}/{MAX_RETRIES}: {e}", temperature=temperature)
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY_SEC)
    
    return fallback_func()

def node_market_screening(state: AgentState) -> dict:
    universe_str = _format_universe(state["market_universe"])
    system_content = SCREENING_SYSTEM_TEMPLATE.format(
        persona_prompt=state["persona_prompt"],
        period_label=state["period_label"],
        memory_context=state.get("memory_context", DEFAULT_MEMORY_CONTEXT)
    )
    human_content = SCREENING_HUMAN_TEMPLATE.format(
        portfolio_json=json.dumps(state["portfolio"]),
        cash=state["cash"],
        universe_str=universe_str
    )
    
    return _invoke_llm_with_retries(
        step_name="screening",
        system_content=system_content,
        human_content=human_content,
        output_model=ScreeningOutput,
        fallback_func=lambda: _safe_screening_fallback(state)
    )


def _node_fundamental_analysis_impl(state: AgentState) -> dict:
    screening = state.get("screening")
    if screening is None:
        state["step_status"] = {**(state.get("step_status", {})), "analysis": "skipped: no screening"}
        return _safe_analysis_fallback()
    if not screening["recheck_tickers"] and not screening["candidate_tickers"]:
        state["step_status"] = {**(state.get("step_status", {})), "analysis": "skipped: no candidates"}
        return _safe_analysis_fallback()

    fundamentals_db = state.get("fundamentals_db", {})
    MAX_NEW_CANDIDATES = 10
    recheck = screening["recheck_tickers"]
    new_candidates = [t for t in screening["candidate_tickers"] if t not in recheck]
    all_candidates = recheck + new_candidates[:MAX_NEW_CANDIDATES]

    detailed = {}
    for t in all_candidates:
        raw = fundamentals_db.get(t, {})
        entry = {}
        for k in _KEY_METRICS:
            if k in raw and raw[k] is not None:
                v = raw[k]
                if isinstance(v, float) and abs(v) > 1_000_000:
                    entry[k] = f"{round(v / 1_000_000, 1)}M"
                else: entry[k] = v
        detailed[t] = entry
    fundamentals_str = json.dumps(detailed, indent=1)

    system_content = ANALYSIS_SYSTEM_TEMPLATE.format(
        persona_prompt=state["persona_prompt"],
        period_label=state["period_label"],
        memory_context=state.get("memory_context", DEFAULT_MEMORY_CONTEXT)
    )
    human_content = ANALYSIS_HUMAN_TEMPLATE.format(
        candidates=all_candidates,
        portfolio_json=json.dumps(state["portfolio"]),
        cash=state["cash"],
        fundamentals_str=fundamentals_str
    )

    return _invoke_llm_with_retries(
        step_name="analyses",
        system_content=system_content,
        human_content=human_content,
        output_model=AnalysisOutput,
        fallback_func=_safe_analysis_fallback
    )


def _node_decision_making_impl(state: AgentState) -> dict:
    analyses_data = state.get("analyses")
    if analyses_data is None or not analyses_data.get("analyses"):
        state["step_status"] = {**(state.get("step_status", {})), "decision": "skipped: no analysis"}
        return _safe_decision_fallback()

    price_data = state.get("price_data", {})
    analysis_summary = []
    for a in analyses_data["analyses"]:
        t = a["ticker"]
        price = price_data.get(t, 100.0)
        held = state["portfolio"].get(t, 0)
        max_buy = int(state["cash"] / price) if price > 0 else 0
        analysis_summary.append({
            "ticker": t, "signal": a["signal"], "conviction": a["conviction"],
            "thesis": a.get("thesis", ""), "price": round(price, 2),
            "held_shares": held, "max_buyable": max_buy,
        })
    analysis_str = json.dumps(analysis_summary, indent=1)

    system_content = DECISION_SYSTEM_TEMPLATE.format(
        persona_prompt=state["persona_prompt"],
        period_label=state["period_label"],
        memory_context=state.get("memory_context", DEFAULT_MEMORY_CONTEXT)
    )
    human_content = DECISION_HUMAN_TEMPLATE.format(
        cash=state["cash"],
        portfolio_json=json.dumps(state["portfolio"]),
        analysis_str=analysis_str
    )

    return _invoke_llm_with_retries(
        step_name="decision",
        system_content=system_content,
        human_content=human_content,
        output_model=DecisionOutput,
        fallback_func=_safe_decision_fallback,
        temperature=DECISION_TEMPERATURE
    )


# ══════════════════════════════════════════════════════════════════════════════
# Graph Construction
# ══════════════════════════════════════════════════════════════════════════════

def _should_continue_after_screening(state):
    s = state.get("screening")
    if s is None: return "end"
    if not (s.get("candidate_tickers", []) + s.get("recheck_tickers", [])): return "end"
    return "analysis"

def _should_continue_after_analysis(state):
    a = state.get("analyses")
    if a is None or not a.get("analyses"): return "end"
    return "decision"

def build_agent_graph():
    graph = StateGraph(AgentState)
    graph.add_node("screening", node_market_screening)
    graph.add_node("analysis", _node_fundamental_analysis_impl)
    graph.add_node("decision", _node_decision_making_impl)
    graph.set_entry_point("screening")
    graph.add_conditional_edges("screening", _should_continue_after_screening, {"analysis": "analysis", "end": END})
    graph.add_conditional_edges("analysis", _should_continue_after_analysis, {"decision": "decision", "end": END})
    graph.add_edge("decision", END)
    return graph.compile()


# ══════════════════════════════════════════════════════════════════════════════
# Backtest Loop
# ══════════════════════════════════════════════════════════════════════════════

INITIAL_CAPITAL = 1_000_000.0

def _execute_trades(executable_orders, hold_orders, portfolio, price_data,
                               memory, period, reverse_ticker_map, rationale):
    """Execute trades, record memory"""
    if executable_orders:
        trade_log = portfolio.apply_trades(executable_orders, price_data)
        for entry in trade_log:
            if entry["status"] == "FILLED":
                executed_order = TradeOrder(
                    ticker=entry["ticker"], action=entry["action"], quantity=entry["quantity"],
                    reasoning=next((o.reasoning for o in executable_orders 
                                   if o.ticker == entry["ticker"] and o.action == entry["action"]), "executed"),
                )
                memory.record_trade(period, executed_order, entry["price"])
        
    memory.record_rationale(period, rationale)


def _process_period(period, portfolio, memory, persona_name, run_idx, n_runs,
                    prices_by_period, fundamentals_by_period, market_universe_by_period,
                    persona_prompt, initial_capital, reverse_ticker_map, agent_graph):
    """Process all decisions and trades for a single period."""
    price_data = prices_by_period[period]
    fundamentals_db = fundamentals_by_period[period]
    market_universe = market_universe_by_period[period]
    
    # Set up logging context
    _lgr = get_logger()
    if _lgr:
        _lgr.set_context(persona=persona_name, period=period, run=run_idx + 1)
    
    # Run agent pipeline
    initial_state: AgentState = {
        "portfolio": dict(portfolio.holdings), "cash": portfolio.cash,
        "market_universe": market_universe, "period_label": period,
        "persona_prompt": persona_prompt, "memory_context": memory.to_prompt_context(),
        "fundamentals_db": fundamentals_db, "price_data": price_data,
    }
    final_state = agent_graph.invoke(initial_state)
    
    # Print step status (skips/errors) with run and period context
    step_status = final_state.get("step_status", {})
    for step, status in step_status.items():
        print(f"ERROR | Run {run_idx + 1}/{n_runs} | {period} | {step.capitalize()}: {status}")
    
    # Print any pipeline errors
    if final_state.get("error"):
        print(f"ERROR | Run {run_idx + 1}/{n_runs} | {period} | Error: {final_state['error']}")
    
    # Process decisions and execute trades
    decision_data = final_state.get("decision")
    if decision_data and decision_data.get("orders"):
        all_orders = [TradeOrder(**o) for o in decision_data["orders"]]
        executable_orders = [o for o in all_orders if o.action in ("BUY", "SELL") and o.quantity > 0]
        hold_orders = [o for o in all_orders if o.action == "HOLD"]
        rationale = decision_data.get("portfolio_rationale", "")
        
        _execute_trades(executable_orders, hold_orders, portfolio, price_data,
                                  memory, period, reverse_ticker_map, rationale)
    else:
        memory.record_rationale(period, "No trades: LLM pipeline returned empty decision.")
    
    # Snapshot and update memory
    pv = portfolio.portfolio_value(price_data)
    portfolio.snapshot(period, price_data)
    memory.update_period_end(period, pv, price_data, initial_capital)
    
    return pv


def run_backtest(persona_name, periods, market_universe_by_period, fundamentals_by_period,
                 prices_by_period, n_runs=3, initial_capital=INITIAL_CAPITAL,
                 reverse_ticker_map=None, final_valuation_prices=None):
    """Execute multi-run backtest for a given persona across multiple periods."""
    persona_prompt = PERSONAS[persona_name]
    all_run_results = []
    agent_graph = build_agent_graph()  # Build once, reuse across all runs and periods

    for run_idx in range(n_runs):
        portfolio = Portfolio(cash=initial_capital)
        memory = AgentMemory()
        period_values = []

        for period in periods:
            print(f"\n{'─'*70}\n  Persona: {persona_name.upper()} | Run {run_idx + 1}/{n_runs} Period: {period}\n{'─'*70}")

            pv = _process_period(period, portfolio, memory, persona_name, run_idx, n_runs,
                                prices_by_period, fundamentals_by_period, market_universe_by_period,
                                persona_prompt, initial_capital, reverse_ticker_map, agent_graph)
            
            period_values.append({"period": period, "portfolio_value": pv})

        # Final valuation at end date (no trading)
        if final_valuation_prices:
            fv = portfolio.portfolio_value(final_valuation_prices)
            period_values.append({"period": "End", "portfolio_value": fv})
            portfolio.snapshot("End", final_valuation_prices)

        perf = portfolio.performance_summary(initial_capital=initial_capital)
        portfolio.save(f"results/portfolios/{persona_name}_run{run_idx + 1}.json", initial_capital=initial_capital)
        all_run_results.append({
            "run": run_idx + 1, "persona": persona_name,
            "period_values": period_values,
            "final_value": period_values[-1]["portfolio_value"] if period_values else 0,
            "trade_count": len(memory.trade_history),
            "performance": perf,
        })

    return all_run_results