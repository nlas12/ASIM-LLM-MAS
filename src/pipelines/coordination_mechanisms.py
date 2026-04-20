"""
Coordination Mechanisms for Multi-Agent Portfolio Management
coordination.py
====================================================
Aggregates and synthesizes trade decisions from multiple agents.
"""

import json
import time
from typing import Optional

from langchain_core.messages import SystemMessage, HumanMessage

import config
from portfolio import TradeOrder
from pipelines.pipeline_utils import (
    DecisionOutput, ProposedTradeOrder,
    make_llm, parse_llm_json,
)
from experiment_logger import get_logger


# ══════════════════════════════════════════════════════════════════════════════
# Prompt Templates
# ══════════════════════════════════════════════════════════════════════════════

REANALYSIS_SYSTEM_TEMPLATE = """{persona_prompt}

Step 5: RE-ANALYSIS with peer context for {period_label}.
{memory_context}

You have peer agent decisions below. Review them but maintain your own philosophy.
For each candidate output:
- ticker, thesis (1 sentence max 20 words), key_metrics (3-5 numbers), signal (STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL), conviction (1-10)
Output ONLY JSON:
{{"analyses":[{{"ticker":"TICK_XX","thesis":"...","key_metrics":{{"pe":18.5,"roe":0.25}},"signal":"BUY","conviction":7}}]}}"""

REANALYSIS_HUMAN_TEMPLATE = """PEER DECISIONS:
{decision_summary}

Portfolio: {portfolio_json}
Cash: ${cash:,.0f}

DATA:
{fundamentals_str}"""

LLM_MANAGER_SYSTEM_TEMPLATE = """You are a neutral portfolio manager synthesizing trade proposals from multiple agents.
1. Find consensus. 2. Resolve conflicts via diversification. 3. Stay within cash. 4. Min 1 order.
Output ONLY JSON:
{"orders":[{"ticker":"TICK_XX","action":"BUY","quantity":100,"reasoning":"max 15 words"}],"portfolio_rationale":"One sentence."}"""

LLM_MANAGER_HUMAN_TEMPLATE = """Holdings: {holdings_json}
Cash: ${cash:,.0f}
Prices: {prices_json}

PROPOSALS:
{proposals_str}

Synthesize into final orders as JSON."""


# ══════════════════════════════════════════════════════════════════════════════
# Coordination Mechanisms
# ══════════════════════════════════════════════════════════════════════════════

def _log(step, sys_prompt, human_prompt, raw_output, parsed, success, error="", temperature=None):
    logger = get_logger()
    if logger:
        logged_temp = temperature if temperature is not None else config.LLM_TEMPERATURE
        logger.log_llm_call(step=step, system_prompt=sys_prompt, human_prompt=human_prompt,
                            raw_output=raw_output or "", parsed_output=parsed,
                            success=success, error=error, temperature=logged_temp)


class CoordinationMechanism:
    name: str = "base"
    def aggregate(self, revised_states, price_data, portfolio) -> tuple[list[TradeOrder], list[str]]:
        """
        Aggregate decisions from multiple agents.
        Returns: (orders, status_messages) where status_messages are for logging.
        """
        raise NotImplementedError


class MajorityVoteCoordination(CoordinationMechanism):
    name = "majority_vote"
    def aggregate(self, revised_states, price_data, portfolio) -> tuple[list[TradeOrder], list[str]]:
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
        return aggregated, []


class AverageSizeCoordination(CoordinationMechanism):
    name = "average_size"
    def aggregate(self, revised_states, price_data, portfolio) -> tuple[list[TradeOrder], list[str]]:
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
        return aggregated, []


class LLMManagerCoordination(CoordinationMechanism):
    name = "llm_manager"
    def aggregate(self, revised_states, price_data, portfolio) -> tuple[list[TradeOrder], list[str]]:
        """
        Use LLM to synthesize multiple agent proposals into final orders.
        Returns: (orders, status_messages)
        """
        status_messages = []
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
            status_messages.append("LLM Manager: no agent proposals. Returning empty.")
            return [], status_messages
        
        for t in portfolio.holdings: proposed_tickers.add(t)
        proposals_str = json.dumps(all_proposals, indent=1)
        relevant_prices = {t: round(price_data[t], 2) for t in proposed_tickers if t in price_data}
        
        system_content = LLM_MANAGER_SYSTEM_TEMPLATE
        human_content = LLM_MANAGER_HUMAN_TEMPLATE.format(
            holdings_json=json.dumps(portfolio.holdings),
            cash=portfolio.cash,
            prices_json=json.dumps(relevant_prices),
            proposals_str=proposals_str
        )
        system = SystemMessage(content=system_content)
        human = HumanMessage(content=human_content)
        
        raw_content = ""
        for attempt in range(config.MAX_RETRIES):
            try:
                llm = make_llm(temperature=config.DECISION_TEMPERATURE)
                response = llm.invoke([system, human])
                raw_content = response.content
                data = parse_llm_json(raw_content)
                decision = DecisionOutput(**data)
                _log("llm_manager", system_content, human_content, raw_content, data, True, temperature=config.DECISION_TEMPERATURE)
                executable = [
                    TradeOrder(ticker=o.ticker, action=o.action, quantity=o.quantity, reasoning=o.reasoning)
                    for o in decision.orders if o.action in ("BUY", "SELL") and o.quantity > 0
                ]
                return executable, status_messages
            except Exception as e:
                _log("llm_manager", system_content, human_content, raw_content,
                     None, False, error=f"Attempt {attempt+1}/{config.MAX_RETRIES}: {e}", temperature=config.DECISION_TEMPERATURE)
                if attempt < config.MAX_RETRIES - 1:
                    status_messages.append(f"LLM Manager attempt {attempt+1} failed: {e}. Retrying...")
                    time.sleep(config.RETRY_DELAY_SEC)
                else:
                    status_messages.append(f"LLM Manager failed after {config.MAX_RETRIES} attempts: {e}. Falling back to majority vote.")
                    print(f"FALLBACK | coordination: llm_manager failed after retry; falling back to majority vote")
                    lgr = get_logger()
                    if lgr:
                        lgr.log_event("FALLBACK", {"step": "llm_manager", "reason": "LLM failure; fell back to majority_vote"})

        return MajorityVoteCoordination().aggregate(revised_states, price_data, portfolio)[0], status_messages


COORDINATION_MECHANISMS = {
    "majority_vote": MajorityVoteCoordination(),
    "average_size": AverageSizeCoordination(),
    "llm_manager": LLMManagerCoordination(),
}
