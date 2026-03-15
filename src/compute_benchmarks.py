"""
Benchmark Portfolio Computation
================================
Standalone benchmark module that fetches WRDS data directly and computes
NAV-based performance metrics for benchmark portfolios:
  1. Market-Cap Weighted (rebalanced each period)
  2. Equal-Weighted (rebalanced each period)
  3. NASDAQ-100 Index (Compustat Global daily index prices)
  4. MSCI World Index (Compustat Global daily index prices)

Strategies 1-2 are index-agnostic: they work with any universe of stocks
(NASDAQ-100, etc.) as long as prices (and market caps for cap-weighted)
are provided per period.

Strategies 3-4 use the official NASDAQ-100 and MSCI World index prices from 
Compustat Global (gvkeyx=000208 for NASDAQ-100, gvkeyx=150066 for MSCI World) 
— no stock-level replication needed.

All metrics match the agent evaluation setup in portfolio.py:
  - Quarterly periods, 2014-01-01 to 2023-12-31
  - NAV starts at 1,000,000
  - r_t = (NAV_t - NAV_{t-1}) / NAV_{t-1}
  - Annualization factor = sqrt(4) for volatility/Sharpe/Sortino
  - rf = 2% annual, MAR = rf for Sortino

Results can be saved to JSON format with the output_json parameter.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd

from data.wrds_data import WRDSDATA


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

INITIAL_NAV = 1_000_000.0
PERIODS_PER_YEAR = 4          # quarterly
RF_ANNUAL = 0.02              # 2% annual risk-free rate
RF_QUARTERLY = RF_ANNUAL / PERIODS_PER_YEAR


# ══════════════════════════════════════════════════════════════════════════════
# WRDS Data Fetching
# ══════════════════════════════════════════════════════════════════════════════

def generate_rebalance_dates(
    start_date: str = "2014-01-01",
    end_date: str = "2023-12-31",
    frequency: str = "quarterly",
) -> list[str]:
    """Generate rebalance dates matching the agent experiment."""
    freq_map = {"quarterly": "QS", "monthly": "MS", "semi-annual": "6MS", "annual": "YS"}
    freq = freq_map.get(frequency)
    if freq is None:
        raise ValueError(f"Unknown frequency: {frequency}")
    dates = pd.date_range(start_date, end_date, freq=freq)
    return [d.strftime("%Y-%m-%d") for d in dates]


def fetch_benchmark_data(
    start_date: str = "2014-01-01",
    end_date: str = "2023-12-31",
    frequency: str = "quarterly",
    window_days: int = 10,
) -> tuple:
    """
    Fetch NASDAQ-100 prices and market caps from WRDS for benchmark computation.

    Connects to WRDS directly, fetches adjusted close prices and shares
    outstanding for all NASDAQ-100 constituents at each rebalance date.
    Market cap is computed as shares_outstanding × close_price.

    Parameters
    ----------
    start_date, end_date : str
        Backtest window in 'YYYY-MM-DD' format.
    frequency : str
        Rebalancing frequency (default: 'quarterly').
    window_days : int
        Lookback window for OHLCV price aggregation.

    Returns
    -------
    tuple of (periods, prices_by_period, market_caps_by_period, final_val_prices)
        periods : list of str
            Period labels ["Period_001", "Period_002", ...].
        prices_by_period : dict
            {period_label: {ticker: adjusted_close_price}}.
        market_caps_by_period : dict
            {period_label: {ticker: market_cap}}.
        final_val_prices : dict
            {ticker: price} at end_date if after last rebalance, else {}.
    """
    rebalance_dates = generate_rebalance_dates(start_date, end_date, frequency)
    periods = [f"Period_{i+1:03d}" for i in range(len(rebalance_dates))]

    print(f"  Benchmark data: {start_date} → {end_date} ({frequency})")
    print(f"  {len(periods)} periods")

    last_rebalance = pd.Timestamp(rebalance_dates[-1]) if rebalance_dates else None
    end_ts = pd.Timestamp(end_date)
    needs_final = last_rebalance is not None and end_ts > last_rebalance

    prices_by_period = {}
    market_caps_by_period = {}
    final_val_prices = {}

    def _extract_prices_and_caps(market_df):
        """Extract {ticker: price} and {ticker: market_cap} from a WRDS DataFrame."""
        price_map = {}
        cap_map = {}
        for _, row in market_df.iterrows():
            ticker = row.get("ticker")
            price = row.get("close")
            shares = row.get("shares_outstanding")
            if pd.notna(ticker) and pd.notna(price):
                t = str(ticker)
                p = float(price)
                price_map[t] = round(p, 2)
                if pd.notna(shares) and float(shares) > 0:
                    cap_map[t] = round(p * float(shares), 2)
        return price_map, cap_map

    with WRDSDATA() as wrds:
        for period, date in zip(periods, rebalance_dates):
            print(f"    {period} ({date})...", end=" ", flush=True)
            market_df = wrds.get_market_data_asof(date, window_days=window_days)
            if market_df.empty:
                print("no data")
                prices_by_period[period] = {}
                market_caps_by_period[period] = {}
                continue

            price_map, cap_map = _extract_prices_and_caps(market_df)
            prices_by_period[period] = price_map
            market_caps_by_period[period] = cap_map
            print(f"{len(price_map)} stocks ({len(cap_map)} with mkt cap)")

        if needs_final:
            print(f"    Final valuation ({end_date})...", end=" ", flush=True)
            market_df = wrds.get_market_data_asof(end_date, window_days=window_days)
            if not market_df.empty:
                for _, row in market_df.iterrows():
                    ticker = row.get("ticker")
                    price = row.get("close")
                    if pd.notna(ticker) and pd.notna(price):
                        final_val_prices[str(ticker)] = round(float(price), 2)
            print(f"{len(final_val_prices)} prices")

    return periods, prices_by_period, market_caps_by_period, final_val_prices


def fetch_msci_world_index(
    start_date: str = "2014-01-01",
    end_date: str = "2023-12-31",
    frequency: str = "quarterly",
    initial_capital: float = INITIAL_NAV,
) -> dict:
    """
    Build an MSCI World benchmark NAV series from the official index prices.

    Fetches daily MSCI World index levels from Compustat Global
    (comp.g_idx_daily, gvkeyx=150066), then for each rebalance date
    finds the closest available price and computes NAV proportionally.

    Parameters
    ----------
    start_date, end_date : str
        Backtest window in 'YYYY-MM-DD' format.
    frequency : str
        Rebalancing frequency (default: 'quarterly').
    initial_capital : float
        Starting portfolio value.

    Returns
    -------
    dict
        period_name → NAV value  (includes "End" key if end_date > last period).
    """
    rebalance_dates = generate_rebalance_dates(start_date, end_date, frequency)
    periods = [f"Period_{i+1:03d}" for i in range(len(rebalance_dates))]

    if not periods:
        return {}

    # Fetch all MSCI World daily prices covering the full range
    # Add buffer before first date to handle weekends / holidays
    fetch_start = (pd.Timestamp(start_date) - pd.Timedelta(days=15)).strftime("%Y-%m-%d")
    with WRDSDATA() as wrds:
        idx_df = wrds.get_msci_world_index_prices(fetch_start, end_date)

    if idx_df.empty:
        print("    WARNING: No MSCI World index data found")
        return {p: initial_capital for p in periods}

    idx_df['datadate'] = pd.to_datetime(idx_df['datadate'])
    idx_df = idx_df.set_index('datadate').sort_index()

    def _nearest_price(date_str: str) -> float | None:
        """Find index price on or just before the given date."""
        ts = pd.Timestamp(date_str)
        mask = idx_df.index <= ts
        if mask.any():
            return float(idx_df.loc[mask, 'price'].iloc[-1])
        return None

    # Map each period to the index level
    result = {}
    base_price = _nearest_price(rebalance_dates[0])
    if base_price is None or base_price <= 0:
        return {p: initial_capital for p in periods}

    for period, date in zip(periods, rebalance_dates):
        p = _nearest_price(date)
        if p is not None and p > 0:
            result[period] = round(initial_capital * (p / base_price), 2)
        else:
            result[period] = round(result.get(periods[periods.index(period)-1], initial_capital), 2)

    # Final valuation if end_date is after last rebalance
    last_rebalance = pd.Timestamp(rebalance_dates[-1])
    if pd.Timestamp(end_date) > last_rebalance:
        p = _nearest_price(end_date)
        if p is not None and p > 0:
            result["End"] = round(initial_capital * (p / base_price), 2)

    return result


def fetch_nasdaq100_index(
    start_date: str = "2014-01-01",
    end_date: str = "2023-12-31",
    frequency: str = "quarterly",
    initial_capital: float = INITIAL_NAV,
) -> dict:
    """
    Build a NASDAQ-100 index benchmark NAV series from the official index prices.

    Fetches daily NASDAQ-100 index levels from Compustat Global
    (comp.g_idx_daily, gvkeyx=000208), then for each rebalance date
    finds the closest available price and computes NAV proportionally.

    Parameters
    ----------
    start_date, end_date : str
        Backtest window in 'YYYY-MM-DD' format.
    frequency : str
        Rebalancing frequency (default: 'quarterly').
    initial_capital : float
        Starting portfolio value.

    Returns
    -------
    dict
        period_name → NAV value  (includes "End" key if end_date > last period).
    """
    rebalance_dates = generate_rebalance_dates(start_date, end_date, frequency)
    periods = [f"Period_{i+1:03d}" for i in range(len(rebalance_dates))]

    if not periods:
        return {}

    # Fetch all NASDAQ-100 daily prices covering the full range
    # Add buffer before first date to handle weekends / holidays
    fetch_start = (pd.Timestamp(start_date) - pd.Timedelta(days=15)).strftime("%Y-%m-%d")
    with WRDSDATA() as wrds:
        idx_df = wrds.get_nasdaq100_index_prices(fetch_start, end_date)

    if idx_df.empty:
        print("    WARNING: No NASDAQ-100 index data found")
        return {p: initial_capital for p in periods}

    idx_df['datadate'] = pd.to_datetime(idx_df['datadate'])
    idx_df = idx_df.set_index('datadate').sort_index()

    def _nearest_price(date_str: str) -> float | None:
        """Find index price on or just before the given date."""
        ts = pd.Timestamp(date_str)
        mask = idx_df.index <= ts
        if mask.any():
            return float(idx_df.loc[mask, 'price'].iloc[-1])
        return None

    # Map each period to the index level
    result = {}
    base_price = _nearest_price(rebalance_dates[0])
    if base_price is None or base_price <= 0:
        return {p: initial_capital for p in periods}

    for period, date in zip(periods, rebalance_dates):
        p = _nearest_price(date)
        if p is not None and p > 0:
            result[period] = round(initial_capital * (p / base_price), 2)
        else:
            result[period] = round(result.get(periods[periods.index(period)-1], initial_capital), 2)

    # Final valuation if end_date is after last rebalance
    last_rebalance = pd.Timestamp(rebalance_dates[-1])
    if pd.Timestamp(end_date) > last_rebalance:
        p = _nearest_price(end_date)
        if p is not None and p > 0:
            result["End"] = round(initial_capital * (p / base_price), 2)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Returns & Metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_returns(nav: np.ndarray) -> np.ndarray:
    """
    Compute period-over-period returns from a NAV series.
        r_t = (NAV_t - NAV_{t-1}) / NAV_{t-1}

    Returns array of length len(nav) - 1.
    """
    return nav[1:] / nav[:-1] - 1.0


def compute_metrics(
    nav: np.ndarray,
    ppy: int = PERIODS_PER_YEAR,
    rf_annual: float = RF_ANNUAL,
) -> dict:
    """
    Compute risk/return metrics from a NAV series.

    Matches the evaluation in portfolio.py exactly:
      - CAGR:       (prod(1+r))^(4/N) - 1
      - Volatility: std_q * sqrt(4)          (ddof=1)
      - Sharpe:     (mean_q - rf_q) / std_q * sqrt(4)
      - Sortino:    (mean_q - rf_q) / dd_q * sqrt(4)   (MAR = rf_q)
      - Max DD:     max( (peak - nav) / peak )
      - Total Return: prod(1+r) - 1
      - Final NAV:  last value in nav array

    Parameters
    ----------
    nav : np.ndarray
        NAV time series (length = n_periods + 1 if final valuation included).
    ppy : int
        Periods per year (4 for quarterly).
    rf_annual : float
        Annual risk-free rate.

    Returns
    -------
    dict with keys: total_return_pct, cagr_pct, volatility_pct,
                    sharpe, sortino, max_drawdown_pct, final_nav, n_periods
    """
    returns = compute_returns(nav)
    n = len(returns)
    if n == 0:
        return {k: 0.0 for k in [
            "total_return_pct", "cagr_pct", "volatility_pct",
            "sharpe", "sortino", "max_drawdown_pct", "final_nav", "n_periods"
        ]}

    rf_q = rf_annual / ppy

    # ── Cumulative return & CAGR ──────────────────────────────────────────
    cum = np.prod(1.0 + returns)
    total_return = cum - 1.0
    cagr = cum ** (ppy / n) - 1.0

    # ── Volatility (annualized) ──────────────────────────────────────────
    vol_q = float(np.std(returns, ddof=1)) if n > 1 else 0.0
    volatility = vol_q * np.sqrt(ppy)

    # ── Sharpe ratio ─────────────────────────────────────────────────────
    mean_excess = float(np.mean(returns)) - rf_q
    sharpe = (mean_excess / vol_q * np.sqrt(ppy)) if vol_q > 0 else 0.0

    # ── Sortino ratio (MAR = rf_q) ───────────────────────────────────────
    downside = returns - rf_q
    downside_neg = np.where(downside < 0, downside, 0.0)
    downside_dev = float(np.sqrt(np.mean(downside_neg ** 2)))
    sortino = (mean_excess / downside_dev * np.sqrt(ppy)) if downside_dev > 0 else 0.0

    # ── Maximum drawdown ─────────────────────────────────────────────────
    peak = np.maximum.accumulate(nav)
    drawdowns = (peak - nav) / peak
    max_dd = float(np.max(drawdowns))

    return {
        "total_return_pct": round(total_return * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "volatility_pct": round(volatility * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "final_nav": round(nav[-1], 2),
        "n_periods": n,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Period-Dict Interface
# ══════════════════════════════════════════════════════════════════════════════
# Builds benchmark NAV from period-keyed price dicts:
#   prices_by_period: dict[str, dict[str, float]]
#     period_name → {ticker: price}

def build_benchmark_series(
    prices_by_period: dict,
    periods: list,
    initial_capital: float = INITIAL_NAV,
    strategy: str = "cap_weighted",
    final_valuation_prices: dict = None,
    market_caps_by_period: dict = None,
) -> dict:
    """
    Build a benchmark NAV series from period-keyed price dicts.

    Both strategies rebalance at each period and are index-agnostic
    (work with any universe — NASDAQ-100, MSCI World, etc.).

    Parameters
    ----------
    prices_by_period : dict
        period_name → {ticker: price}
    periods : list of str
        Ordered period labels (e.g. ["Period_001", "Period_002", ...]).
    initial_capital : float
        Starting portfolio value.
    strategy : str
        "cap_weighted" — market-cap weighted, rebalanced each period.
        "equal_weight" — equal-dollar allocation, rebalanced each period.
    final_valuation_prices : dict, optional
        {ticker: price} for final valuation after last trading period.
    market_caps_by_period : dict, optional
        period_name → {ticker: market_cap}. Required for "cap_weighted".

    Returns
    -------
    dict
        period_name → NAV value  (includes "End" key if final_valuation_prices given).
    """
    if not periods:
        return {}

    if strategy == "cap_weighted":
        if not market_caps_by_period:
            raise ValueError("cap_weighted strategy requires market_caps_by_period")

        result = {}
        holdings = {}
        current_value = initial_capital
        for period in periods:
            pp = prices_by_period.get(period, {})
            caps = market_caps_by_period.get(period, {})
            valid = {t: p for t, p in pp.items() if p and p > 0 and t in caps and caps[t] > 0}
            if not valid:
                result[period] = round(current_value, 2)
                continue
            # Mark-to-market existing holdings
            if holdings:
                current_value = sum(holdings.get(t, 0) * pp.get(t, 0) for t in holdings if pp.get(t, 0))
            # Compute market-cap weights and rebalance
            total_cap = sum(caps[t] for t in valid)
            weights = {t: caps[t] / total_cap for t in valid}
            holdings = {t: (current_value * weights[t]) / valid[t] for t in valid}
            total = sum(holdings[t] * valid[t] for t in holdings)
            result[period] = round(total, 2)
            current_value = total
        if final_valuation_prices and holdings:
            result["End"] = round(
                sum(holdings.get(t, 0) * final_valuation_prices.get(t, 0) for t in holdings), 2
            )
        return result

    elif strategy == "equal_weight":
        result = {}
        holdings = {}
        current_value = initial_capital
        for period in periods:
            pp = prices_by_period.get(period, {})
            valid = {t: p for t, p in pp.items() if p and p > 0}
            if not valid:
                result[period] = round(current_value, 2)
                continue
            if holdings:
                current_value = sum(holdings.get(t, 0) * valid.get(t, 0) for t in holdings)
            per_stock = current_value / len(valid)
            holdings = {t: per_stock / p for t, p in valid.items()}
            total = sum(holdings[t] * valid[t] for t in holdings)
            result[period] = round(total, 2)
            current_value = total
        if final_valuation_prices and holdings:
            result["End"] = round(
                sum(holdings.get(t, 0) * final_valuation_prices.get(t, 0) for t in holdings), 2
            )
        return result

    else:
        raise ValueError(f"Unknown benchmark strategy: {strategy}")


# ══════════════════════════════════════════════════════════════════════════════
# Standalone Benchmark Runner (fetches its own WRDS data)
# ══════════════════════════════════════════════════════════════════════════════

def run_benchmarks_standalone(
    start_date: str = "2014-01-01",
    end_date: str = "2023-12-31",
    frequency: str = "quarterly",
    initial_capital: float = INITIAL_NAV,
    output_json: str = None,
) -> dict:
    """
    Fully self-contained benchmark computation.

    Connects to WRDS, fetches NASDAQ-100 prices and market caps at each
    rebalance date, and computes cap-weighted + equal-weight benchmark
    NAV series with performance metrics, plus NASDAQ-100 and MSCI World indices.

    Parameters
    ----------
    start_date, end_date : str
        Backtest window in 'YYYY-MM-DD' format.
    frequency : str
        Rebalancing frequency (default: 'quarterly').
    initial_capital : float
        Starting portfolio value (default: 1,000,000).
    output_json : str, optional
        Path to save results as JSON. If None, results are not saved.

    Returns
    -------
    dict with keys:
        "benchmarks" : dict
            {"cap_weighted": {period: NAV}, "equal_weight": {period: NAV}, 
             "nasdaq100_index": {period: NAV}, "msci_world": {period: NAV}}
        "metrics" : dict
            {"cap_weighted": {...}, "equal_weight": {...}, 
             "nasdaq100_index": {...}, "msci_world": {...}}
        "periods" : list of str
        "prices_by_period" : dict
            {period: {ticker: price}}
    """
    print("=" * 60)
    print("Standalone Benchmark Computation")
    print("=" * 60)

    # ── Fetch WRDS data (NASDAQ-100) ─────────────────────────────────────
    periods, prices_by_period, market_caps_by_period, final_val_prices = fetch_benchmark_data(
        start_date=start_date,
        end_date=end_date,
        frequency=frequency,
    )

    # ── Build benchmark NAV series ───────────────────────────────────────
    cw_nav = build_benchmark_series(
        prices_by_period, periods,
        initial_capital=initial_capital,
        strategy="cap_weighted",
        final_valuation_prices=final_val_prices,
        market_caps_by_period=market_caps_by_period,
    )
    ew_nav = build_benchmark_series(
        prices_by_period, periods,
        initial_capital=initial_capital,
        strategy="equal_weight",
        final_valuation_prices=final_val_prices,
    )

    # ── NASDAQ-100 index benchmark ──────────────────────────────────────
    print("  Fetching NASDAQ-100 index...")
    nasdaq_nav = fetch_nasdaq100_index(
        start_date=start_date,
        end_date=end_date,
        frequency=frequency,
        initial_capital=initial_capital,
    )
    print(f"    {len([p for p in nasdaq_nav if p != 'End'])} periods")

    # ── MSCI World index benchmark ───────────────────────────────────────
    print("  Fetching MSCI World index...")
    msci_nav = fetch_msci_world_index(
        start_date=start_date,
        end_date=end_date,
        frequency=frequency,
        initial_capital=initial_capital,
    )
    print(f"    {len([p for p in msci_nav if p != 'End'])} periods")

    benchmarks = {
        "cap_weighted": cw_nav,
        "equal_weight": ew_nav,
        "nasdaq100_index": nasdaq_nav,
        "msci_world": msci_nav,
    }

    # ── Compute metrics ──────────────────────────────────────────────────
    metrics = {}
    for name, nav_dict in benchmarks.items():
        nav_values = [nav_dict[p] for p in periods if p in nav_dict]
        if "End" in nav_dict:
            nav_values.append(nav_dict["End"])
        nav_arr = np.array(nav_values, dtype=float)
        metrics[name] = compute_metrics(nav_arr)

    # ── Print summary ────────────────────────────────────────────────────
    print(f"\n{'Benchmark':<20} {'CAGR':>8} {'Vol':>8} {'Sharpe':>8} {'Final NAV':>14}")
    print("-" * 60)
    for name in ["cap_weighted", "equal_weight", "nasdaq100_index", "msci_world"]:
        m = metrics[name]
        label = name.replace("_", " ").title()
        print(f"  {label:<18} {m['cagr_pct']:>7.2f}% {m['volatility_pct']:>7.2f}% "
              f"{m['sharpe']:>8.3f} ${m['final_nav']:>13,.2f}")
    print()

    result = {
        "benchmarks": benchmarks,
        "metrics": metrics,
        "periods": periods,
        "prices_by_period": prices_by_period,
    }

    # ── Save to JSON if specified ────────────────────────────────────────
    if output_json:
        print(f"Saving results to {output_json}...")
        try:
            with open(output_json, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Results saved successfully.")
        except Exception as e:
            print(f"WARNING: Could not save JSON to {output_json}: {e}")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Standalone execution
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Example: Run benchmarks and save to JSON
    output_file = Path(__file__).resolve().parent / "benchmark_results.json"
    result = run_benchmarks_standalone(output_json=str(output_file))
    print("Done. Keys returned:", list(result.keys()))
