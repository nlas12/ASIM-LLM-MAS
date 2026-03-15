"""
Multi-Agent Investment Pipeline — Phase 1 Refactor
multi_agent_pipeline.py
====================================================
Extends single_agent_pipeline.py with the CI coordination layer.
"""

import json
import re
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional
from collections import Counter

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from experiment_logger import get_logger
from personas import PERSONAS
from portfolio import Portfolio
from single_agent_pipeline import (
    AgentState, AgentMemory, DecisionOutput, TradeOrder, AnalysisOutput,
    build_agent_graph, make_llm, parse_llm_json,
    _node_fundamental_analysis_impl, _node_decision_making_impl,
    _KEY_METRICS, INITIAL_CAPITAL, MAX_RETRIES, RETRY_DELAY_SEC,
    _safe_analysis_fallback, _safe_decision_fallback,
    LLM_TEMPERATURE, DECISION_TEMPERATURE,
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
        print("  ✗ Re-analysis skipped: no candidates.")
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

    system = SystemMessage(content=f"""{state["persona_prompt"]}

Step 5: RE-ANALYSIS with peer context for {state["period_label"]}.
{memory_context}

You have peer agent decisions below. Review them but maintain your own philosophy.
For each candidate output:
- ticker, thesis (1 sentence max 20 words), key_metrics (3-5 numbers), signal (STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL), conviction (1-10)
Output ONLY JSON:
{{"analyses":[{{"ticker":"TICK_XX","thesis":"...","key_metrics":{{"pe":18.5,"roe":0.25}},"signal":"BUY","conviction":7}}]}}""")
    human = HumanMessage(content=f"""PEER DECISIONS:
{decision_summary}

Portfolio: {json.dumps(state["portfolio"])}
Cash: ${state["cash"]:,.0f}

DATA:
{fundamentals_str}""")

    raw_content = ""
    for attempt in range(MAX_RETRIES):
        try:
            llm = make_llm()
            response = llm.invoke([system, human])
            raw_content = response.content
            data = parse_llm_json(raw_content)
            analysis = AnalysisOutput(**data)
            _log("reanalysis", system.content, human.content, raw_content, data, True)
            return {"analyses": analysis.model_dump()}
        except Exception as e:
            _log("reanalysis", system.content, human.content, raw_content,
                 None, False, error=f"Attempt {attempt+1}/{MAX_RETRIES}: {e}")
            if attempt < MAX_RETRIES - 1:
                print(f"  ⚠ Re-analysis attempt {attempt+1} failed: {e}. Retrying...")
                time.sleep(RETRY_DELAY_SEC)
            else:
                print(f"  ✗ Re-analysis failed after {MAX_RETRIES} attempts: {e}. Keeping Pass 1 analysis.")
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
# Coordination Mechanisms (Step 7)
# ══════════════════════════════════════════════════════════════════════════════

class CoordinationMechanism:
    name: str = "base"
    def aggregate(self, revised_states, price_data, portfolio) -> list[TradeOrder]:
        raise NotImplementedError

class MajorityVoteCoordination(CoordinationMechanism):
    name = "majority_vote"
    def aggregate(self, revised_states, price_data, portfolio):
        n_agents = len(revised_states)
        ticker_votes = {}
        for state in revised_states.values():
            decision = state.get("decision")
            if decision is None or not decision.get("orders"): continue
            for order in decision["orders"]:
                ticker_votes.setdefault(order["ticker"], []).append(order)
        aggregated = []
        for ticker, orders in ticker_votes.items():
            buy_orders = [o for o in orders if o["action"] == "BUY"]
            sell_orders = [o for o in orders if o["action"] == "SELL"]
            if len(buy_orders) > n_agents / 2:
                avg_qty = max(1, int(sum(o["quantity"] for o in buy_orders) / len(buy_orders)))
                price = price_data.get(ticker, 0.0)
                if price > 0 and avg_qty * price <= portfolio.cash:
                    aggregated.append(TradeOrder(ticker=ticker, action="BUY", quantity=avg_qty,
                                                 reasoning=f"Majority BUY ({len(buy_orders)}/{n_agents} agents)"))
            elif len(sell_orders) > n_agents / 2:
                avg_qty = max(1, int(sum(o["quantity"] for o in sell_orders) / len(sell_orders)))
                available = portfolio.holdings.get(ticker, 0)
                qty = min(avg_qty, available)
                if qty > 0:
                    aggregated.append(TradeOrder(ticker=ticker, action="SELL", quantity=qty,
                                                 reasoning=f"Majority SELL ({len(sell_orders)}/{n_agents} agents)"))
        return aggregated

class AverageSizeCoordination(CoordinationMechanism):
    name = "average_size"
    def aggregate(self, revised_states, price_data, portfolio):
        ticker_orders = {}
        for state in revised_states.values():
            decision = state.get("decision")
            if decision is None or not decision.get("orders"): continue
            for order in decision["orders"]:
                ticker_orders.setdefault(order["ticker"], []).append(order)
        aggregated = []
        for ticker, orders in ticker_orders.items():
            buys = [o["quantity"] for o in orders if o["action"] == "BUY"]
            sells = [o["quantity"] for o in orders if o["action"] == "SELL"]
            net = len(buys) - len(sells)
            if net > 0:
                avg_qty = max(1, int(sum(buys) / len(buys)))
                price = price_data.get(ticker, 0.0)
                if price > 0 and avg_qty * price <= portfolio.cash:
                    aggregated.append(TradeOrder(ticker=ticker, action="BUY", quantity=avg_qty,
                                                 reasoning=f"Net BUY ({len(buys)} vs {len(sells)}), avg qty"))
            elif net < 0:
                avg_qty = max(1, int(sum(sells) / len(sells)))
                available = portfolio.holdings.get(ticker, 0)
                qty = min(avg_qty, available)
                if qty > 0:
                    aggregated.append(TradeOrder(ticker=ticker, action="SELL", quantity=qty,
                                                 reasoning=f"Net SELL ({len(sells)} vs {len(buys)}), avg qty"))
        return aggregated

class LLMManagerCoordination(CoordinationMechanism):
    name = "llm_manager"
    def aggregate(self, revised_states, price_data, portfolio):
        all_proposals = []
        proposed_tickers = set()
        for persona, state in revised_states.items():
            decision = state.get("decision")
            if decision is None or not decision.get("orders"): continue
            compact_orders = [{"ticker": o["ticker"], "action": o["action"],
                               "quantity": o["quantity"], "reasoning": o["reasoning"]} for o in decision["orders"]]
            for o in decision["orders"]: proposed_tickers.add(o["ticker"])
            all_proposals.append({"persona": persona, "orders": compact_orders})
        if not all_proposals:
            print("  ✗ LLM Manager: no agent proposals. Returning empty.")
            return []
        for t in portfolio.holdings: proposed_tickers.add(t)
        proposals_str = json.dumps(all_proposals, indent=1)
        relevant_prices = {t: round(price_data[t], 2) for t in proposed_tickers if t in price_data}
        system = SystemMessage(content="""You are a neutral portfolio manager synthesizing trade proposals from multiple agents.
1. Find consensus. 2. Resolve conflicts via diversification. 3. Stay within cash. 4. Min 1 order.
Output ONLY JSON:
{"orders":[{"ticker":"TICK_XX","action":"BUY","quantity":100,"reasoning":"max 15 words"}],"portfolio_rationale":"One sentence."}""")
        human = HumanMessage(content=f"""Holdings: {json.dumps(portfolio.holdings)}
Cash: ${portfolio.cash:,.0f}
Prices: {json.dumps(relevant_prices)}

PROPOSALS:
{proposals_str}

Synthesize into final orders as JSON.""")
        raw_content = ""
        for attempt in range(MAX_RETRIES):
            try:
                llm = make_llm(temperature=DECISION_TEMPERATURE)
                response = llm.invoke([system, human])
                raw_content = response.content
                data = parse_llm_json(raw_content)
                decision = DecisionOutput(**data)
                _log("llm_manager", system.content, human.content, raw_content, data, True, temperature=DECISION_TEMPERATURE)
                return decision.orders
            except Exception as e:
                _log("llm_manager", system.content, human.content, raw_content,
                     None, False, error=f"Attempt {attempt+1}/{MAX_RETRIES}: {e}", temperature=DECISION_TEMPERATURE)
                if attempt < MAX_RETRIES - 1:
                    print(f"  ⚠ LLM Manager attempt {attempt+1} failed: {e}. Retrying...")
                    time.sleep(RETRY_DELAY_SEC)
                else:
                    print(f"  ✗ LLM Manager failed after {MAX_RETRIES} attempts. Falling back to majority vote.")
        return MajorityVoteCoordination().aggregate(revised_states, price_data, portfolio)

COORDINATION_MECHANISMS = {
    "majority_vote": MajorityVoteCoordination(),
    "average_size": AverageSizeCoordination(),
    "llm_manager": LLMManagerCoordination(),
}


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
    reverse_ticker_map: dict[str, str] | None = None,
    final_valuation_prices: dict[str, float] | None = None,
    workers: int = 1,
) -> list[dict]:
    coordinator = COORDINATION_MECHANISMS[coordination]
    all_run_results = []

    def resolve(t):
        if reverse_ticker_map and t in reverse_ticker_map: return f"{t} ({reverse_ticker_map[t]})"
        return t
    def resolve_short(t):
        if reverse_ticker_map and t in reverse_ticker_map: return reverse_ticker_map[t]
        return t

    for run_idx in range(n_runs):
        print(f"\n{'='*70}")
        print(f"Multi-Agent | Coordination: {coordination.upper()} | Run {run_idx + 1}/{n_runs}")
        print(f"Personas: {', '.join(p.upper() for p in persona_names)}")
        print(f"{'='*70}")

        portfolio = Portfolio(cash=initial_capital)
        agent_memories = {name: AgentMemory() for name in persona_names}
        period_values = []

        for period in periods:
            print(f"\n{'─'*70}")
            print(f"  Period: {period}")
            print(f"{'─'*70}")

            price_data = prices_by_period[period]
            fundamentals_db = fundamentals_by_period[period]
            market_universe = market_universe_by_period[period]

            # ══ PASS 1 ════════════════════════════════════════════════════
            print(f"\n  ┌─ PASS 1: Independent Agent Decisions{' (parallel)' if workers > 1 else ''}")
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

            for persona in persona_names:
                state = first_pass_states[persona]
                print(f"  │\n  ├─ {persona.upper()}")
                screening = state.get("screening")
                if screening:
                    candidates = screening.get("candidate_tickers", [])
                    print(f"  │  [Screening] {len(candidates)} candidates: {', '.join(resolve_short(t) for t in candidates[:8])}{'...' if len(candidates) > 8 else ''}")
                analyses = state.get("analyses")
                if analyses and analyses.get("analyses"):
                    for a in sorted(analyses["analyses"], key=lambda a: a.get("conviction", 0), reverse=True)[:5]:
                        print(f"  │  [Analysis] {resolve_short(a['ticker']):<12s} {a['signal']:<12s} conviction={a['conviction']}")
                else: print(f"  │  [Analysis] empty (fallback)")
                decision = state.get("decision")
                if decision and decision.get("orders"):
                    for o in decision["orders"]:
                        print(f"  │    {o['action']:<5s} {resolve_short(o['ticker']):<12s} qty={o['quantity']:<6d} | {o['reasoning'][:80]}")
                else: print(f"  │  [Decision] no orders (fallback)")
            print(f"  └─ Pass 1 complete")

            # ══ Step 4 ════════════════════════════════════════════════════
            decision_summary = build_decision_summary(first_pass_states)
            any_orders = any(s.get("decision", {}).get("orders") for s in first_pass_states.values())
            print(f"\n  [Step 4] Decision summary {'built' if any_orders else 'empty (no orders in Pass 1)'}.")

            # ══ PASS 2 ════════════════════════════════════════════════════
            print(f"\n  ┌─ PASS 2: Peer-Informed Revision{' (parallel)' if workers > 1 else ''}")
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

            for persona in persona_names:
                revised_state = second_pass_states[persona]
                print(f"  │\n  ├─ {persona.upper()} (revised)")
                decision = revised_state.get("decision")
                if decision and decision.get("orders"):
                    for o in decision["orders"]:
                        print(f"  │    {o['action']:<5s} {resolve_short(o['ticker']):<12s} qty={o['quantity']:<6d} | {o['reasoning'][:80]}")
                else: print(f"  │  [Revised Decision] no orders (holding)")
            print(f"  └─ Pass 2 complete")

            # ══ Step 7: Coordination ══════════════════════════════════════
            print(f"\n  [Step 7] COORDINATION ({coordination.upper()})")
            final_orders = coordinator.aggregate(second_pass_states, price_data, portfolio)
            print(f"  Aggregated orders: {len(final_orders)}")
            for o in final_orders:
                print(f"    {o.action:<5s} {resolve(o.ticker):<30s} qty={o.quantity:<6d} | {o.reasoning}")

            # ══ Step 8: Execute ═══════════════════════════════════════════
            trade_log = portfolio.apply_trades(final_orders, price_data)
            for entry in trade_log:
                if entry["status"] == "FILLED":
                    # Record with actual executed quantity (may differ from order due to cash/holdings caps)
                    executed_order = TradeOrder(
                        ticker=entry["ticker"],
                        action=entry["action"],
                        quantity=entry["quantity"],
                        reasoning=next((o.reasoning for o in final_orders if o.ticker == entry["ticker"] and o.action == entry["action"]), "coordinated"),
                    )
                    for memory in agent_memories.values():
                        memory.record_trade(period, executed_order, entry["price"])
            print(f"\n  [Step 8] EXECUTION")
            for entry in trade_log:
                if entry["status"] == "FILLED":
                    val = entry.get("cost", entry.get("proceeds", 0))
                    print(f"    {entry['action']:<5s} {resolve(entry['ticker']):<30s} {entry['quantity']:>6d} @ ${entry['price']:>10.2f} = ${val:>14,.2f}")
                else:
                    print(f"    SKIP  {resolve(entry['ticker'])}: {entry.get('reason', '')}")
            if not trade_log: print(f"    (no trades this period)")

            pv = portfolio.portfolio_value(price_data)
            equity = sum(s * price_data.get(t, 0.0) for t, s in portfolio.holdings.items())
            period_values.append({"period": period, "portfolio_value": pv})
            portfolio.snapshot(period, price_data)
            for memory in agent_memories.values():
                memory.update_period_end(period, pv, price_data, initial_capital)
            for persona in persona_names:
                decision = second_pass_states[persona].get("decision")
                agent_memories[persona].record_rationale(
                    period, decision.get("portfolio_rationale", "") if decision else "No revised decision produced.")

            print(f"\n  Portfolio Value: ${pv:,.2f} (Cash: ${portfolio.cash:,.2f} | Equity: ${equity:,.2f})")
            if portfolio.holdings:
                print(f"  Holdings:")
                for t, shares in sorted(portfolio.holdings.items()):
                    price = price_data.get(t, 0.0)
                    print(f"    {resolve_short(t):<12s} {shares:>6d} shares @ ${price:>10.2f} = ${shares * price:>14,.2f}")

        # ── Final valuation at end date (no trading, mark-to-market only) ─
        if final_valuation_prices:
            fv = portfolio.portfolio_value(final_valuation_prices)
            period_values.append({"period": "End", "portfolio_value": fv})
            portfolio.snapshot("End", final_valuation_prices)
            print(f"\n{'─'*70}")
            print(f"  Final Valuation (end date, no trading)")
            print(f"  Portfolio: ${fv:,.2f}")
            print(f"{'─'*70}")

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