"""
Portfolio execution & performance analytics for the backtesting pipeline.

Tracks holdings, cash, snapshots (NAV per period), and computes
standard risk-adjusted return metrics on quarterly return series.
Includes portfolio composition analysis: allocation tracking, position sizing,
and concentration metrics (Herfindahl index, max position %, diversification).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════════════════════════════
# Trade order schema (duplicated minimally to avoid circular imports)
# ══════════════════════════════════════════════════════════════════════════════

class TradeOrder(BaseModel):
    """A single buy or sell order."""
    ticker: str
    action: str = Field(description="BUY or SELL")
    quantity: int = Field(description="Number of shares. Must be > 0.", gt=0)
    reasoning: str = Field(description="One sentence justification.")


# ══════════════════════════════════════════════════════════════════════════════
# Portfolio
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Portfolio:
    """
    Simple portfolio that tracks anonymous-ticker holdings and cash.

    Provides:
    - ``apply_trades``: execute BUY/SELL orders against current state
    - ``portfolio_value``: mark-to-market NAV
    - ``get_portfolio_allocation``: compute position weights and allocation percentages
    - ``get_concentration_metrics``: analyze portfolio balancing (Herfindahl, diversification)
    - ``snapshot``: record end-of-period NAV with allocation and concentration data
    """

    holdings: dict[str, int] = field(default_factory=dict)
    cash: float = 1_000_000.0
    snapshots: list[dict] = field(default_factory=list)

    # ── trade execution ───────────────────────────────────────────────────

    def apply_trades(
        self, orders: list[TradeOrder], price_data: dict
    ) -> list[dict]:
        """
        Execute orders against portfolio.
        Returns structured trade log (for memory recording).
        """
        log: list[dict] = []
        for order in orders:
            price = price_data.get(order.ticker, 0.0)
            if price <= 0:
                log.append({
                    "status": "SKIP",
                    "ticker": order.ticker,
                    "reason": "no price data",
                })
                continue

            if order.action == "BUY":
                cost = price * order.quantity
                if cost > self.cash:
                    affordable = int(self.cash / price)
                    if affordable == 0:
                        log.append({
                            "status": "SKIP",
                            "ticker": order.ticker,
                            "action": "BUY",
                            "reason": "insufficient cash",
                        })
                        continue
                    order = TradeOrder(
                        ticker=order.ticker,
                        action="BUY",
                        quantity=affordable,
                        reasoning=order.reasoning,
                    )
                    cost = price * affordable

                self.cash -= cost
                self.holdings[order.ticker] = (
                    self.holdings.get(order.ticker, 0) + order.quantity
                )
                log.append({
                    "status": "FILLED",
                    "ticker": order.ticker,
                    "action": "BUY",
                    "quantity": order.quantity,
                    "price": price,
                    "cost": cost,
                })

            elif order.action == "SELL":
                available = self.holdings.get(order.ticker, 0)
                qty = min(order.quantity, available)
                if qty == 0:
                    log.append({
                        "status": "SKIP",
                        "ticker": order.ticker,
                        "action": "SELL",
                        "reason": "no position",
                    })
                    continue
                proceeds = price * qty
                self.cash += proceeds
                self.holdings[order.ticker] = available - qty
                if self.holdings[order.ticker] == 0:
                    del self.holdings[order.ticker]
                log.append({
                    "status": "FILLED",
                    "ticker": order.ticker,
                    "action": "SELL",
                    "quantity": qty,
                    "price": price,
                    "proceeds": proceeds,
                })

        return log

    # ── valuation ─────────────────────────────────────────────────────────

    def portfolio_value(self, price_data: dict) -> float:
        """Mark-to-market NAV (cash + equity)."""
        equity = sum(
            shares * price_data.get(ticker, 0.0)
            for ticker, shares in self.holdings.items()
        )
        return self.cash + equity

    def get_portfolio_allocation(self, price_data: dict) -> dict:
        """
        Compute allocation percentages for each position.

        Returns
        -------
        dict
            ``{ticker: {value, weight_pct}, ..., "cash": {value, weight_pct}}``
        """
        nav = self.portfolio_value(price_data)
        if nav <= 0:
            return {"cash": {"value": self.cash, "weight_pct": 100.0}}

        allocation = {}
        for ticker, shares in self.holdings.items():
            value = shares * price_data.get(ticker, 0.0)
            weight_pct = (value / nav) * 100
            allocation[ticker] = {
                "value": round(value, 2),
                "weight_pct": round(weight_pct, 2),
            }

        cash_pct = (self.cash / nav) * 100
        allocation["cash"] = {
            "value": round(self.cash, 2),
            "weight_pct": round(cash_pct, 2),
        }
        return allocation

    def get_concentration_metrics(self, price_data: dict) -> dict:
        """
        Compute portfolio concentration and balancing metrics.

        Returns
        -------
        dict
            Keys: ``num_holdings``, ``herfindahl_index`` (0-1), ``max_position_pct``,
                  ``cash_pct``, ``equity_allocated_pct``.
        """
        nav = self.portfolio_value(price_data)
        if nav <= 0:
            return {
                "num_holdings": 0,
                "herfindahl_index": 0.0,
                "max_position_pct": 0.0,
                "cash_pct": 100.0,
                "equity_allocated_pct": 0.0,
            }

        # Count non-zero holdings
        num_holdings = len(self.holdings)

        # Compute position weights
        weights = []
        for ticker, shares in self.holdings.items():
            value = shares * price_data.get(ticker, 0.0)
            weight = value / nav
            weights.append(weight)

        # Herfindahl index (sum of squared weights): 0 = perfectly balanced, 1 = all in one position
        herfindahl = float(np.sum(np.array(weights) ** 2)) if weights else 0.0

        # Max position size
        max_position = max(weights) * 100 if weights else 0.0

        # Cash and equity allocation
        equity_value = nav - self.cash
        equity_pct = (equity_value / nav) * 100 if nav > 0 else 0.0
        cash_pct = (self.cash / nav) * 100 if nav > 0 else 0.0

        return {
            "num_holdings": num_holdings,
            "herfindahl_index": round(herfindahl, 4),
            "max_position_pct": round(max_position, 2),
            "cash_pct": round(cash_pct, 2),
            "equity_allocated_pct": round(equity_pct, 2),
        }

    # ── snapshots ─────────────────────────────────────────────────────────

    def snapshot(self, period: str, price_data: dict) -> dict:
        """
        Record end-of-period portfolio state including allocation and concentration metrics.

        Parameters
        ----------
        period : str
            Period label, e.g. ``"Period_003"``.
        price_data : dict
            ``{ticker: price}`` for the current period.

        Returns
        -------
        dict
            The snapshot that was appended (for convenience). Includes:
            - period, nav, cash, equity, holdings
            - allocation: position weights and values
            - concentration: Herfindahl index, max position %, portfolio balancing metrics
        """
        nav = self.portfolio_value(price_data)
        equity = sum(
            shares * price_data.get(t, 0.0)
            for t, shares in self.holdings.items()
        )
        allocation = self.get_portfolio_allocation(price_data)
        concentration = self.get_concentration_metrics(price_data)

        entry = {
            "period": period,
            "nav": nav,
            "cash": self.cash,
            "equity": equity,
            "holdings": dict(self.holdings),
            "allocation": allocation,
            "concentration": concentration,
        }
        self.snapshots.append(entry)
        return entry

    def snapshots_df(self) -> pd.DataFrame:
        """
        Return snapshots as a DataFrame.

        Includes allocation percentages and concentration metrics for each period.
        Drops the nested holdings dict for a clean table view.
        """
        if not self.snapshots:
            return pd.DataFrame()
        df = pd.DataFrame(self.snapshots)
        # Drop the nested holdings dict for a clean table
        return df.drop(columns=["holdings"], errors="ignore")

    # ── persistence ───────────────────────────────────────────────────────

    def save(
        self,
        path: str
    ) -> None:
        """
        Save snapshots and performance metrics to a JSON file.

        The output contains:
        - ``snapshots``: list of per-period dicts (period, nav, cash, equity, holdings)

        Parameters
        ----------
        path : str
            File path for the JSON output (directories created automatically).
        initial_capital : float, optional
        """
        import json, os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        payload = {
            "snapshots": self.snapshots,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
