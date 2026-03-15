"""
Portfolio execution & performance analytics for the backtesting pipeline.

Tracks holdings, cash, snapshots (NAV per period), and computes
standard risk-adjusted return metrics on quarterly return series.
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
    - ``snapshot``: record end-of-period NAV for later analytics
    - ``performance_summary``: compute CAGR, vol, Sharpe, Sortino, max-DD
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

    # ── snapshots ─────────────────────────────────────────────────────────

    def snapshot(self, period: str, price_data: dict) -> dict:
        """
        Record end-of-period portfolio state.

        Parameters
        ----------
        period : str
            Period label, e.g. ``"Period_003"``.
        price_data : dict
            ``{ticker: price}`` for the current period.

        Returns
        -------
        dict
            The snapshot that was appended (for convenience).
        """
        nav = self.portfolio_value(price_data)
        equity = sum(
            shares * price_data.get(t, 0.0)
            for t, shares in self.holdings.items()
        )
        entry = {
            "period": period,
            "nav": nav,
            "cash": self.cash,
            "equity": equity,
            "holdings": dict(self.holdings),
        }
        self.snapshots.append(entry)
        return entry

    # ── performance analytics ─────────────────────────────────────────────

    # Risk-free rate: 2 % p.a.  →  ~0.5 % per quarter
    _RF_ANNUAL: float = 0.02
    _PERIODS_PER_YEAR: int = 4  # quarterly

    def performance_summary(
        self,
        initial_capital: float | None = None,
        rf_annual: float | None = None,
    ) -> dict:
        """
        Compute key performance metrics from the recorded snapshots.

        All returns are derived from mark-to-market NAV (cash + equity).

        Metrics
        -------
        - **CAGR** (annualized return):
          ``(∏(1+r_t))^(4/N) − 1``
        - **Volatility** (annualized):
          ``σ_q · √4``
        - **Sharpe Ratio**:
          ``(r̄_q − r_f,q) / σ_q · √4``
        - **Sortino Ratio**:
          ``(r̄_q − r_f,q) / σ_d,q · √4``
          where ``σ_d,q`` is the downside deviation below ``r_f,q``.
        - **Maximum Drawdown**:
          ``max_t( (Peak_t − NAV_t) / Peak_t )``

        Parameters
        ----------
        initial_capital : float, optional
            Starting NAV (default: first snapshot NAV if not provided; 
            falls back to 1_000_000).
        rf_annual : float, optional
            Annual risk-free rate (default: 0.02).

        Returns
        -------
        dict  with keys ``cagr``, ``volatility``, ``sharpe``, ``sortino``,
              ``max_drawdown``, ``total_return``, ``n_periods``.
        """
        if not self.snapshots:
            return {
                "total_return_pct": 0.0,
                "annualized_return_pct": 0.0,
                "annualized_volatility_pct": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown_pct": 0.0,
                "final_portfolio_value": 0.0,
                "n_periods": 0,
            }

        rf = rf_annual if rf_annual is not None else self._RF_ANNUAL
        ppy = self._PERIODS_PER_YEAR
        rf_q = rf / ppy  # quarterly risk-free rate

        nav_series = np.array([s["nav"] for s in self.snapshots], dtype=float)

        n_returns = len(nav_series) - 1  # number of quarterly return intervals

        # ── quarterly returns ─────────────────────────────────────────────
        returns = nav_series[1:] / nav_series[:-1] - 1.0  # length = n_returns

        # ── CAGR ─────────────────────────────────────────────────────────
        cum_return = np.prod(1.0 + returns)
        cagr = cum_return ** (ppy / n_returns) - 1.0 if n_returns > 0 else 0.0

        # ── total return ──────────────────────────────────────────────────
        total_return = cum_return - 1.0

        # ── volatility (annualized) ──────────────────────────────────────
        vol_q = float(np.std(returns, ddof=1)) if n_returns > 1 else 0.0
        volatility = vol_q * math.sqrt(ppy)

        # ── Sharpe ratio ─────────────────────────────────────────────────
        mean_excess = float(np.mean(returns)) - rf_q
        sharpe = (mean_excess / vol_q * math.sqrt(ppy)) if vol_q > 0 else 0.0

        # ── Sortino ratio ────────────────────────────────────────────────
        # MAR (Minimum Acceptable Return) = rf_q = r_f / 4  (quarterly).
        # Downside deviation is computed only from returns below MAR;
        # returns >= MAR contribute 0 to the semi-variance.
        downside = returns - rf_q
        downside_neg = np.where(downside < 0, downside, 0.0)
        downside_dev = float(np.sqrt(np.mean(downside_neg ** 2)))
        sortino = (
            (mean_excess / downside_dev * math.sqrt(ppy))
            if downside_dev > 0
            else 0.0
        )

        # ── maximum drawdown ─────────────────────────────────────────────
        peak = np.maximum.accumulate(nav_series)
        drawdowns = (peak - nav_series) / peak
        max_drawdown = float(np.max(drawdowns))

        return {
            "total_return_pct": round(float(total_return) * 100, 2),
            "annualized_return_pct": round(float(cagr) * 100, 2),
            "annualized_volatility_pct": round(float(volatility) * 100, 2),
            "sharpe_ratio": round(float(sharpe), 3),
            "sortino_ratio": round(float(sortino), 3),
            "max_drawdown_pct": round(float(max_drawdown) * 100, 2),
            "final_portfolio_value": round(float(nav_series[-1]), 2),
            "n_periods": n_returns,
        }

    def snapshots_df(self) -> pd.DataFrame:
        """Return snapshots as a DataFrame."""
        if not self.snapshots:
            return pd.DataFrame()
        df = pd.DataFrame(self.snapshots)
        # Drop the nested holdings dict for a clean table
        return df.drop(columns=["holdings"], errors="ignore")

    # ── persistence ───────────────────────────────────────────────────────

    def save(
        self,
        path: str,
        initial_capital: float | None = None,
        rf_annual: float | None = None,
    ) -> None:
        """
        Save snapshots and performance metrics to a JSON file.

        The output contains:
        - ``snapshots``: list of per-period dicts (period, nav, cash, equity, holdings)
        - ``metrics``: the full ``performance_summary`` dict

        Parameters
        ----------
        path : str
            File path for the JSON output (directories created automatically).
        initial_capital : float, optional
            Passed through to ``performance_summary``.
        rf_annual : float, optional
            Passed through to ``performance_summary``.
        """
        import json, os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        payload = {
            "snapshots": self.snapshots,
            "metrics": self.performance_summary(
                initial_capital=initial_capital,
                rf_annual=rf_annual,
            ),
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=str)


# ══════════════════════════════════════════════════════════════════════════════
# Standalone helper — works from plain NAV series (e.g. benchmark indices)
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(
    period_values: list[dict],
    initial_capital: float = 1_000_000.0,
) -> dict:
    """
    Compute performance metrics from a list of
    ``{"period": ..., "portfolio_value": float}`` dicts.

    Useful for benchmark series that don't have a Portfolio object.
    Internally creates a temporary Portfolio, populates snapshots, and
    delegates to ``Portfolio.performance_summary``.
    """
    if not period_values:
        return {}
    p = Portfolio(cash=0.0)
    for pv in period_values:
        p.snapshots.append({
            "period": pv["period"],
            "nav": pv["portfolio_value"],
            "cash": 0.0,
            "equity": pv["portfolio_value"],
            "holdings": {},
        })
    return p.performance_summary(initial_capital=initial_capital)
