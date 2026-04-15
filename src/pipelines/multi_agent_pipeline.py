"""
Multi-Agent Investment Pipeline — Phase 1 Refactor
multi_agent_pipeline.py
====================================================
Extends single_agent_pipeline.py with the CI coordination layer.
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage

from experiment_logger import get_logger
from personas import PERSONAS
from portfolio import Portfolio
from pipelines.coordination_mechanisms import COORDINATION_MECHANISMS, REANALYSIS_SYSTEM_TEMPLATE, REANALYSIS_HUMAN_TEMPLATE
from pipelines.pipeline_utils import (
    AgentState, AgentMemory, TradeOrder, AnalysisOutput,
    make_llm, parse_llm_json,
    _KEY_METRICS, INITIAL_CAPITAL, MAX_RETRIES, RETRY_DELAY_SEC,
    LLM_TEMPERATURE,
)
from pipelines.single_agent_pipeline import (
    build_agent_graph, _node_decision_making_impl,
)


# ══════════════════════════════════════════════════════════════════════════════
# Decision Summary (Step 4)
# ══════════════════════════════════════════════════════════════════════════════

def build_decision_summary(agent_results: dict[str, dict]) -> str:
    summary = {"period": None, "agent_decisions": []}
    for persona, state in agent_results.items():
        decision = state.get("decision")
        if decision is None or not decision.get("orders"): continue
        summary["period"] = state.get("period_label", "unknown")
        compact_orders = [{"ticker": o["ticker"], "action": o["action"],
                           "quantity": o["quantity"], "reasoning": o["reasoning"]}
                          for o in decision["orders"]]
        top_signals = []
        analyses_data = state.get("analyses", {})
        if analyses_data and analyses_data.get("analyses"):
            for a in analyses_data["analyses"]:
                if a["signal"] in ("STRONG_BUY", "BUY", "STRONG_SELL", "SELL"):
                    top_signals.append({"ticker": a["ticker"], "signal": a["signal"], "conviction": a["conviction"]})
        summary["agent_decisions"].append({"persona": persona, "orders": compact_orders, "top_signals": top_signals[:8]})
    return json.dumps(summary, indent=1)


# ══════════════════════════════════════════════════════════════════════════════
# Re-Analysis Node (Step 5)
# ══════════════════════════════════════════════════════════════════════════════

def _log(step, sys_prompt, human_prompt, raw_output, parsed, success, error="", temperature=None):
    logger = get_logger()
    if logger:
        logged_temp = temperature if temperature is not None else LLM_TEMPERATURE
        logger.log_llm_call(step=step, system_prompt=sys_prompt, human_prompt=human_prompt,
                            raw_output=raw_output or "", parsed_output=parsed,
                            success=success, error=error, temperature=logged_temp)

def node_reanalysis_with_peers(state: AgentState, fundamentals_db: dict, decision_summary: str) -> dict:
    memory_context = state.get("memory_context", "FIRST PERIOD.")
    screening = state.get("screening")
    if screening:
        recheck = screening["recheck_tickers"]
        new_candidates = [t for t in screening["candidate_tickers"] if t not in recheck]
        all_candidates = recheck + new_candidates[:10]
    else:
        all_candidates = [s["ticker"] for s in state["market_universe"][:12]]
    if not all_candidates:
        return {}

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

    system_content = REANALYSIS_SYSTEM_TEMPLATE.format(
        persona_prompt=state["persona_prompt"],
        period_label=state["period_label"],
        memory_context=memory_context
    )
    human_content = REANALYSIS_HUMAN_TEMPLATE.format(
        decision_summary=decision_summary,
        portfolio_json=json.dumps(state["portfolio"]),
        cash=state["cash"],
        fundamentals_str=fundamentals_str
    )
    system = SystemMessage(content=system_content)
    human = HumanMessage(content=human_content)

    raw_content = ""
    for attempt in range(MAX_RETRIES):
        try:
            llm = make_llm()
            response = llm.invoke([system, human])
            raw_content = response.content
            data = parse_llm_json(raw_content)
            analysis = AnalysisOutput(**data)
            _log("reanalysis", system_content, human_content, raw_content, data, True)
            return {"analyses": analysis.model_dump()}
        except Exception as e:
            _log("reanalysis", system_content, human_content, raw_content,
                 None, False, error=f"Attempt {attempt+1}/{MAX_RETRIES}: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY_SEC)
    return {}

def node_revised_decision(state: AgentState, price_data: dict) -> dict:
    return _node_decision_making_impl(state, price_data)


# ══════════════════════════════════════════════════════════════════════════════
# Pass 2 Graph
# ══════════════════════════════════════════════════════════════════════════════

def _should_redecide_after_reanalysis(state: AgentState) -> str:
    analyses = state.get("analyses")
    if analyses is None: return "end"
    if not analyses.get("analyses"): return "end"
    return "redecision"

def build_pass2_graph(fundamentals_db, price_data, decision_summary):
    g = StateGraph(AgentState)
    g.add_node("reanalysis", lambda state: node_reanalysis_with_peers(state, fundamentals_db, decision_summary))
    g.add_node("redecision", lambda state: node_revised_decision(state, price_data))
    g.set_entry_point("reanalysis")
    g.add_conditional_edges("reanalysis", _should_redecide_after_reanalysis, {"redecision": "redecision", "end": END})
    g.add_edge("redecision", END)
    return g.compile()


# ══════════════════════════════════════════════════════════════════════════════
# Period Processing (Multi-Agent)
# ══════════════════════════════════════════════════════════════════════════════

def _process_period_multiagent(period, portfolio, agent_memories, persona_names, run_idx, n_runs,
                               prices_by_period, fundamentals_by_period, market_universe_by_period,
                               coordinator, workers, initial_capital):
    """Execute Pass 1, Pass 2, coordination, and trading for a single period."""
    price_data = prices_by_period[period]
    fundamentals_db = fundamentals_by_period[period]
    market_universe = market_universe_by_period[period]

    # ══ PASS 1 ════════════════════════════════════════════════════════
    first_pass_states = {}

    def _run_pass1(persona):
        _lgr = get_logger()
        if _lgr: _lgr.set_context(persona=persona, period=period, run=run_idx+1)
        memory = agent_memories[persona]
        app = build_agent_graph(fundamentals_db, price_data)
        initial_state: AgentState = {
            "portfolio": dict(portfolio.holdings), "cash": portfolio.cash,
            "market_universe": market_universe, "period_label": period,
            "persona_prompt": PERSONAS[persona], "memory_context": memory.to_prompt_context(),
        }
        return persona, app.invoke(initial_state)

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_run_pass1, p) for p in persona_names]
            for future in as_completed(futures):
                persona, state = future.result()
                first_pass_states[persona] = state
    else:
        for persona in persona_names:
            p, state = _run_pass1(persona)
            first_pass_states[p] = state

    # ══ Step 4 ════════════════════════════════════════════════════
    decision_summary = build_decision_summary(first_pass_states)

    # ══ PASS 2 ════════════════════════════════════════════════════
    second_pass_states = {}

    def _run_pass2(persona):
        _lgr = get_logger()
        if _lgr: _lgr.set_context(persona=persona, period=period, run=run_idx+1)
        re_app = build_pass2_graph(fundamentals_db, price_data, decision_summary)
        return persona, re_app.invoke(first_pass_states[persona])

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_run_pass2, p) for p in persona_names]
            for future in as_completed(futures):
                persona, revised_state = future.result()
                second_pass_states[persona] = revised_state
    else:
        for persona in persona_names:
            p, revised_state = _run_pass2(persona)
            second_pass_states[p] = revised_state

    # ══ Step 7: Coordination ══════════════════════════════════════
    final_orders, coord_status = coordinator.aggregate(second_pass_states, price_data, portfolio)
    for status_msg in coord_status:
        print(f"ERROR | Run {run_idx + 1}/{n_runs} | {period} | COORDINATOR: {status_msg}")

    # ══ Step 8: Execute ═══════════════════════════════════════════
    trade_log = portfolio.apply_trades(final_orders, price_data)
    for entry in trade_log:
        if entry["status"] == "FILLED":
            executed_order = TradeOrder(
                ticker=entry["ticker"],
                action=entry["action"],
                quantity=entry["quantity"],
                reasoning=next((o.reasoning for o in final_orders if o.ticker == entry["ticker"] and o.action == entry["action"]), "coordinated"),
            )
            for memory in agent_memories.values():
                memory.record_trade(period, executed_order, entry["price"])

    # Update portfolio and memory
    pv = portfolio.portfolio_value(price_data)
    portfolio.snapshot(period, price_data)
    for memory in agent_memories.values():
        memory.update_period_end(period, pv, price_data, initial_capital)
    for persona in persona_names:
        decision = second_pass_states[persona].get("decision")
        agent_memories[persona].record_rationale(
            period, decision.get("portfolio_rationale", "") if decision else "No revised decision produced.")

    return pv


# ══════════════════════════════════════════════════════════════════════════════
# Multi-Agent Backtest Loop
# ══════════════════════════════════════════════════════════════════════════════

def run_multiagent_backtest(
    persona_names: list[str],
    periods: list[str],
    market_universe_by_period: dict,
    fundamentals_by_period: dict,
    prices_by_period: dict,
    coordination: str = "majority_vote",
    n_runs: int = 3,
    initial_capital: float = INITIAL_CAPITAL,
    final_valuation_prices: dict[str, float] | None = None,
    workers: int = 1,
) -> list[dict]:
    """Coordinate multi-agent runs. Delegates period execution to _process_period_multiagent."""
    coordinator = COORDINATION_MECHANISMS[coordination]
    all_run_results = []

    for run_idx in range(n_runs):
        print(f"Multi-Agent | Coordination: {coordination.upper()} | Run {run_idx + 1}/{n_runs}")

        portfolio = Portfolio(cash=initial_capital)
        agent_memories = {name: AgentMemory() for name in persona_names}
        period_values = []

        for period in periods:
            pv = _process_period_multiagent(
                period, portfolio, agent_memories, persona_names, run_idx, n_runs,
                prices_by_period, fundamentals_by_period, market_universe_by_period,
                coordinator, workers, initial_capital
            )
            period_values.append({"period": period, "portfolio_value": pv})

        # ── Final valuation at end date (no trading, mark-to-market only) ─
        if final_valuation_prices:
            fv = portfolio.portfolio_value(final_valuation_prices)
            period_values.append({"period": "End", "portfolio_value": fv})
            portfolio.snapshot("End", final_valuation_prices)

        perf = portfolio.performance_summary(initial_capital=initial_capital)
        portfolio.save(
            f"results/portfolios/{coordination}_run{run_idx + 1}.json",
            initial_capital=initial_capital,
        )
        all_run_results.append({
            "run": run_idx + 1, "coordination": coordination, "personas": persona_names,
            "period_values": period_values,
            "final_value": period_values[-1]["portfolio_value"] if period_values else 0,
            "performance": perf,
        })

    return all_run_results