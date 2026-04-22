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

import config
from experiment_logger import get_logger, with_logger
from personas import PERSONAS
from portfolio import Portfolio, TradeOrder
from pipelines.coordination_mechanisms import COORDINATION_MECHANISMS, REANALYSIS_SYSTEM_TEMPLATE, REANALYSIS_HUMAN_TEMPLATE
from pipelines.pipeline_utils import (
    AgentState, AgentMemory, AnalysisOutput,
    make_llm, parse_llm_json,
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

def _log(step, sys_prompt, human_prompt, raw_output, parsed, success, error="", temperature=None, logger=None):
    if logger is None:
        logger = get_logger()
    if logger:
        logged_temp = temperature if temperature is not None else config.LLM_TEMPERATURE
        logger.log_llm_call(step=step, system_prompt=sys_prompt, human_prompt=human_prompt,
                            raw_output=raw_output or "", parsed_output=parsed,
                            success=success, error=error, temperature=logged_temp)

def node_reanalysis_with_peers(state: AgentState, fundamentals_db: dict, decision_summary: str, logger=None) -> dict:
    memory_context = state.get("memory_context", "FIRST PERIOD.")
    screening = state.get("screening")
    if screening:
        recheck = screening["recheck_tickers"]
        new_candidates = [t for t in screening["candidate_tickers"] if t not in recheck]
        all_candidates = recheck + new_candidates[:config.MAX_NEW_CANDIDATES]
    else:
        all_candidates = [s["ticker"] for s in state["market_universe"][:12]]
    if not all_candidates:
        return {}

    detailed = {}
    for t in all_candidates:
        raw = fundamentals_db.get(t, {})
        entry = {}
        for k in config.KEY_METRICS:
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
    for attempt in range(config.MAX_RETRIES):
        try:
            llm = make_llm()
            response = llm.invoke([system, human])
            raw_content = response.content
            data = parse_llm_json(raw_content)
            analysis = AnalysisOutput(**data)
            _log("reanalysis", system_content, human_content, raw_content, data, True, logger=logger)
            return {"analyses": analysis.model_dump()}
        except Exception as e:
            _log("reanalysis", system_content, human_content, raw_content,
                 None, False, error=f"Attempt {attempt+1}/{config.MAX_RETRIES}: {e}", logger=logger)
            if attempt < config.MAX_RETRIES - 1:
                time.sleep(config.RETRY_DELAY_SEC)
    # All retries exhausted — announce fallback and return empty analyses
    state.setdefault("step_status", {})["reanalysis"] = "fallback: reanalysis failed after retry"
    period = state.get("period_label", "?")
    print(f"FALLBACK | {period} | reanalysis: LLM re-analysis failed after retry")
    lgr = logger or get_logger()
    if lgr:
        lgr.log_event("FALLBACK", {"step": "reanalysis", "reason": "LLM failure"}, period=period)
    return {"analyses": {"analyses": []}}

def node_revised_decision(state: AgentState, logger=None) -> dict:
    return _node_decision_making_impl(state, logger=logger)


# ══════════════════════════════════════════════════════════════════════════════
# Pass 2 Graph
# ══════════════════════════════════════════════════════════════════════════════

def _should_redecide_after_reanalysis(state: AgentState) -> str:
    analyses = state.get("analyses")
    if analyses is None: return "end"
    if not analyses.get("analyses"): return "end"
    return "redecision"

def build_pass2_graph(fundamentals_db, decision_summary, logger=None):
    g = StateGraph(AgentState)
    g.add_node("reanalysis", lambda state: node_reanalysis_with_peers(state, fundamentals_db, decision_summary, logger=logger))
    g.add_node("redecision", lambda state: node_revised_decision(state, logger=logger))
    g.set_entry_point("reanalysis")
    g.add_conditional_edges("reanalysis", _should_redecide_after_reanalysis, {"redecision": "redecision", "end": END})
    g.add_edge("redecision", END)
    return g.compile()


# ══════════════════════════════════════════════════════════════════════════════
# Period Processing (Multi-Agent)
# ══════════════════════════════════════════════════════════════════════════════

def _process_period_multiagent(period, portfolio, agent_memories, persona_names, run_idx, n_runs,
                               prices_by_period, fundamentals_by_period, market_universe_by_period,
                               coordinator, workers, initial_capital, agent_loggers=None, coordinator_logger=None):
    """Execute Pass 1, Pass 2, coordination, and trading for a single period.
    
    Args:
        agent_loggers: dict mapping persona name -> ExperimentLogger instance (one per agent)
        coordinator_logger: logger for coordination events
    """
    price_data = prices_by_period[period]
    fundamentals_db = fundamentals_by_period[period]
    market_universe = market_universe_by_period[period]

    # ══ PASS 1 ════════════════════════════════════════════════════════
    first_pass_states = {}

    def _run_pass1(persona):
        memory = agent_memories[persona]
        agent_logger = agent_loggers.get(persona) if agent_loggers else None
        with with_logger(agent_logger):
            if agent_logger:
                agent_logger.set_context(pipeline="multi_agent", persona=persona,
                                         period=period, run=run_idx + 1)
            app = build_agent_graph(logger=agent_logger)
            initial_state: AgentState = {
                "portfolio": dict(portfolio.holdings), "cash": portfolio.cash,
                "market_universe": market_universe, "period_label": period,
                "persona_prompt": PERSONAS[persona], "memory_context": memory.to_prompt_context(),
                "price_data": price_data, "fundamentals_db": fundamentals_db,
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

    # ══ Step 4: Decision Summary ════════════════════════════════════
    decision_summary = build_decision_summary(first_pass_states)
    if coordinator_logger:
        coordinator_logger.log_event("decision_summary", {"summary": json.loads(decision_summary)}, persona="coordinator", period=period)

    # ══ PASS 2 ════════════════════════════════════════════════════
    second_pass_states = {}

    def _run_pass2(persona):
        agent_logger = agent_loggers.get(persona) if agent_loggers else None
        with with_logger(agent_logger):
            if agent_logger:
                agent_logger.set_context(pipeline="multi_agent", persona=persona,
                                         period=period, run=run_idx + 1)
            re_app = build_pass2_graph(fundamentals_db, decision_summary, logger=agent_logger)
            pass2_state = first_pass_states[persona].copy()
            pass2_state["price_data"] = price_data
            return persona, re_app.invoke(pass2_state)

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
    with with_logger(coordinator_logger):
        if coordinator_logger:
            coordinator_logger.set_context(pipeline="multi_agent", persona="coordinator",
                                           period=period, run=run_idx + 1)
        final_orders, coord_status = coordinator.aggregate(second_pass_states, price_data, portfolio)
    if coordinator_logger:
        coordinator_logger.log_event("coordination", {
            "mechanism": coordinator.__class__.__name__,
            "final_orders_count": len(final_orders),
            "coordination_notes": coord_status
        }, persona="coordinator", period=period)
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
    n_runs: int = 1,
    initial_capital: float = config.INITIAL_CAPITAL,
    final_valuation_prices: dict[str, float] | None = None,
    workers: int = 1,
    experiment_root_dir: str | None = None,
) -> list[dict]:
    """Coordinate multi-agent runs. Delegates period execution to _process_period_multiagent."""
    import os
    from experiment_logger import ExperimentLogger
    from run_metadata import write_run_metadata

    coordinator = COORDINATION_MECHANISMS[coordination]
    all_run_results = []

    for run_idx in range(n_runs):
        # Create run-specific folder structure if experiment_root_dir provided
        if experiment_root_dir:
            run_folder = os.path.join(experiment_root_dir, f"run{run_idx + 1}")
            portfolio_folder = os.path.join(run_folder, "portfolios")
            os.makedirs(portfolio_folder, exist_ok=True)
        else:
            run_folder = None
            portfolio_folder = "results/portfolios"
            os.makedirs(portfolio_folder, exist_ok=True)
        
        if run_folder:
            write_run_metadata(run_folder, mode="multi_agent",
                               coordination=coordination, personas=persona_names,
                               run_idx=run_idx + 1, n_runs=n_runs)

        # Create per-agent logger instances
        agent_loggers = {}
        for persona in persona_names:
            if run_folder:
                log_path = os.path.join(run_folder, f"experiment_log_{persona}.json")
            else:
                log_path = f"results/experiment_log_{persona}.json"
            agent_loggers[persona] = ExperimentLogger(log_path=log_path, model=config.MODEL_NAME)

        # Create coordinator logger
        if run_folder:
            coordinator_log_path = os.path.join(run_folder, f"experiment_log_coordinator.json")
        else:
            coordinator_log_path = "results/experiment_log_coordinator.json"
        coordinator_logger = ExperimentLogger(log_path=coordinator_log_path, model=config.MODEL_NAME)
        
        print(f"Multi-Agent | Coordination: {coordination.upper()} | Run {run_idx + 1}/{n_runs}")

        portfolio = Portfolio(cash=initial_capital)
        agent_memories = {name: AgentMemory() for name in persona_names}
        period_values = []

        for period in periods:
            print(f"\n Started Run {run_idx + 1}/{n_runs} | Period {period}/{periods}")
            pv = _process_period_multiagent(
                period, portfolio, agent_memories, persona_names, run_idx, n_runs,
                prices_by_period, fundamentals_by_period, market_universe_by_period,
                coordinator, workers, initial_capital, agent_loggers=agent_loggers, 
                coordinator_logger=coordinator_logger
            )
            period_values.append({"period": period, "portfolio_value": pv})

        # ── Final valuation at end date (no trading, mark-to-market only) ─
        if final_valuation_prices:
            fv = portfolio.portfolio_value(final_valuation_prices)
            period_values.append({"period": "End", "portfolio_value": fv})
            portfolio.snapshot("End", final_valuation_prices)

        portfolio_filename = os.path.join(portfolio_folder, f"{coordination}.json")
        portfolio.save(portfolio_filename)
        
        # Save all agent loggers
        for persona, logger in agent_loggers.items():
            logger.save()
            logger.print_summary()
        
        # Save coordinator logger
        coordinator_logger.save()
        coordinator_logger.print_summary()
        
        final_value = period_values[-1]["portfolio_value"] if period_values else initial_capital
        all_run_results.append({
            "run": run_idx + 1, "coordination": coordination, "personas": persona_names,
            "period_values": period_values,
            "final_value": final_value,
        })

    return all_run_results