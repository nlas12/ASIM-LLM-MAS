"""
Single-Agent Investment Pipeline
single_agent_pipeline.py
================================================================
"""
import json
import time
from typing import Optional

from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage

from portfolio import Portfolio
from personas import PERSONAS
from pipelines.pipeline_utils import (
    AgentState, AgentMemory, DecisionOutput, TradeOrder, AnalysisOutput, ScreeningOutput,
    make_llm, parse_llm_json, _log,
    MAX_RETRIES, RETRY_DELAY_SEC, DECISION_TEMPERATURE, DEFAULT_MEMORY_CONTEXT, INITIAL_CAPITAL,
    MAX_NEW_CANDIDATES, _KEY_METRICS,
)
from experiment_logger import get_logger


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
# Node Implementations
# ══════════════════════════════════════════════════════════════════════════════
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
# Node Implementations
# ══════════════════════════════════════════════════════════════════════════════

def _format_universe(market_universe):
    if not market_universe: return "[]"
    return json.dumps(market_universe, indent=1)

def _invoke_llm_with_retries(step_name: str, system_content: str, human_content: str,
                              output_model: type, fallback_func: callable,
                              temperature: Optional[float] = None, logger=None) -> dict:
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
            _log(step_name, system_content, human_content, raw_content, data, True, temperature=temperature, logger=logger)
            return {step_name: result.model_dump()}
        except Exception as e:
            _log(step_name, system_content, human_content, raw_content,
                 None, False, error=f"Attempt {attempt+1}/{MAX_RETRIES}: {e}", temperature=temperature, logger=logger)
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY_SEC)
    
    return fallback_func()

def node_market_screening(state: AgentState, logger=None) -> dict:
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
        fallback_func=lambda: _safe_screening_fallback(state),
        logger=logger
    )


def _node_fundamental_analysis_impl(state: AgentState, logger=None) -> dict:
    screening = state.get("screening")
    if screening is None:
        state["step_status"] = {**(state.get("step_status", {})), "analysis": "skipped: no screening"}
        return _safe_analysis_fallback()
    if not screening["recheck_tickers"] and not screening["candidate_tickers"]:
        state["step_status"] = {**(state.get("step_status", {})), "analysis": "skipped: no candidates"}
        return _safe_analysis_fallback()

    fundamentals_db = state.get("fundamentals_db", {})
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
        fallback_func=_safe_analysis_fallback,
        logger=logger
    )


def _node_decision_making_impl(state: AgentState, logger=None) -> dict:
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
        temperature=DECISION_TEMPERATURE,
        logger=logger
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

def build_agent_graph(logger=None):
    # Create wrapper functions that pass the logger to nodes
    def screening_node(state):
        return node_market_screening(state, logger=logger)
    def analysis_node(state):
        return _node_fundamental_analysis_impl(state, logger=logger)
    def decision_node(state):
        return _node_decision_making_impl(state, logger=logger)
    
    graph = StateGraph(AgentState)
    graph.add_node("screening", screening_node)
    graph.add_node("analysis", analysis_node)
    graph.add_node("decision", decision_node)
    graph.set_entry_point("screening")
    graph.add_conditional_edges("screening", _should_continue_after_screening, {"analysis": "analysis", "end": END})
    graph.add_conditional_edges("analysis", _should_continue_after_analysis, {"decision": "decision", "end": END})
    graph.add_edge("decision", END)
    return graph.compile()


# ══════════════════════════════════════════════════════════════════════════════
# Backtest Loop
# ══════════════════════════════════════════════════════════════════════════════

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
                 reverse_ticker_map=None, final_valuation_prices=None,
                 experiment_root_dir=None, parent_experiment_root_dir=None):
    """Execute multi-run backtest for a given persona across multiple periods.
    
    Args:
        experiment_root_dir: Root folder for this experiment (creates single_agent_MMDD.HHMM.SS if None)
        parent_experiment_root_dir: If provided, save to parent's run folder instead (for nested multi-agent calls)
    """
    import os
    from experiment_logger import ExperimentLogger
    
    persona_prompt = PERSONAS[persona_name]
    all_run_results = []

    for run_idx in range(n_runs):
        # Determine folder structure based on context
        if parent_experiment_root_dir:
            # Running as part of multi-agent - use parent's run structure
            run_folder = os.path.join(parent_experiment_root_dir, f"run{run_idx + 1}")
            portfolio_folder = os.path.join(run_folder, "portfolios")
            log_path = os.path.join(run_folder, f"experiment_log_{persona_name}.json")
        elif experiment_root_dir:
            # Standalone single-agent with custom root
            run_folder = os.path.join(experiment_root_dir, f"run{run_idx + 1}")
            portfolio_folder = os.path.join(run_folder, "portfolios")
            log_path = os.path.join(run_folder, f"experiment_log_{persona_name}.json")
        else:
            # Default fallback
            portfolio_folder = "results/portfolios"
            log_path = "results/experiment_log.json"
        
        os.makedirs(portfolio_folder, exist_ok=True)
        
        # Create a new logger instance for this run
        logger = ExperimentLogger(log_path=log_path)
        agent_graph = build_agent_graph(logger=logger)  # Build graph with logger for this run
        
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

        portfolio_filename = os.path.join(portfolio_folder, f"{persona_name}.json")
        portfolio.save(portfolio_filename)
        
        # Save the logger for this run
        logger.save()
        logger.print_summary()
        
        final_value = period_values[-1]["portfolio_value"] if period_values else initial_capital

        all_run_results.append({
            "run": run_idx + 1, "persona": persona_name,
            "period_values": period_values,
            "final_value": final_value,
            "trade_count": len(memory.trade_history),
        })

    return all_run_results