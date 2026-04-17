"""
Data Loader — WRDS Integration via existing WRDSDATA class
data_loader.py
============================================================
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hashlib
import warnings
import math
from typing import Optional

import pandas as pd
import numpy as np

from data.wrds_data import WRDSDATA


# ══════════════════════════════════════════════════════════════════════════════
# Anonymization
# ══════════════════════════════════════════════════════════════════════════════

def anonymize_ticker(real_ticker: str, salt: str = "asim2526") -> str:
    h = hashlib.sha256(f"{salt}{real_ticker}".encode()).hexdigest()[:6].upper()
    return f"TICK_{h}"

def build_ticker_map(real_tickers: list[str]) -> dict[str, str]:
    return {t: anonymize_ticker(t) for t in real_tickers}


# ══════════════════════════════════════════════════════════════════════════════
# Rebalance Date Generation
# ══════════════════════════════════════════════════════════════════════════════

def generate_rebalance_dates(start_date: str, end_date: str, frequency: str = "quarterly") -> list[str]:
    freq_map = {"quarterly": "QS", "monthly": "MS", "semi-annual": "6MS", "annual": "YS"}
    freq = freq_map.get(frequency)
    if freq is None:
        raise ValueError(f"Unknown frequency '{frequency}'. Choose from: {list(freq_map.keys())}")
    dates = pd.date_range(start_date, end_date, freq=freq)
    return [d.strftime("%Y-%m-%d") for d in dates]


# ══════════════════════════════════════════════════════════════════════════════
# Core Data Fetching
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_period_data(wrds, date, window_days=10):
    market_df = wrds.get_market_data_asof(date, window_days=window_days)
    if market_df.empty:
        return pd.DataFrame(), {}, []
    market_df = market_df.dropna(subset=["ticker"])
    price_map = {}
    if "close" in market_df.columns:
        for _, row in market_df.iterrows():
            if pd.notna(row.get("close")) and pd.notna(row.get("ticker")):
                price_map[row["ticker"]] = float(row["close"])
    ticker_list = market_df["ticker"].dropna().unique().tolist()
    return market_df, price_map, ticker_list

def _fetch_prices_only(wrds, date, ticker_map, window_days=10):
    market_df = wrds.get_market_data_asof(date, window_days=window_days)
    if market_df.empty:
        return {}
    prices = {}
    if "close" in market_df.columns:
        for _, row in market_df.iterrows():
            ticker = row.get("ticker")
            if pd.notna(ticker) and pd.notna(row.get("close")) and ticker in ticker_map:
                prices[ticker_map[ticker]] = round(float(row["close"]), 2)
    return prices


def _compute_derived_ratios(row: pd.Series) -> dict:
    """Compute derived financial ratios from a single row of Compustat + CRSP data."""
    ratios = {}

    def safe_div(num, den):
        if den is None or den == 0 or num is None: return None
        return num / den

    def safe_float(val):
        if pd.isna(val): return None
        return float(val)

    assets = safe_float(row.get("assets"))
    liabilities = safe_float(row.get("liabilities"))
    revenue = safe_float(row.get("revenue"))
    net_income = safe_float(row.get("net_income"))
    cash = safe_float(row.get("cash_equivalents"))
    current_assets = safe_float(row.get("current_assets"))
    current_liabilities = safe_float(row.get("current_liabilities"))
    long_term_debt = safe_float(row.get("long_term_debt"))
    shares = safe_float(row.get("shares_outstanding"))
    close_price = safe_float(row.get("close"))
    operating_income = safe_float(row.get("operating_income"))

    equity = (assets - liabilities) if (assets and liabilities) else None

    # ── Standard ratios ───────────────────────────────────────────────────
    ratios["roe"] = safe_div(net_income, equity)
    ratios["roa"] = safe_div(net_income, assets)
    ratios["current_ratio"] = safe_div(current_assets, current_liabilities)
    ratios["debt_to_equity"] = safe_div(liabilities, equity)
    ratios["debt_ratio"] = safe_div(liabilities, assets)
    ratios["net_income_margin"] = safe_div(net_income, revenue)

    # ── Market cap & valuation ────────────────────────────────────────────
    market_cap = None
    if shares and close_price:
        market_cap = shares * close_price
        ratios["market_cap"] = market_cap
        if net_income and net_income > 0:
            annual_earnings = net_income * 4
            eps = annual_earnings / shares
            ratios["pe_ratio"] = close_price / eps if eps > 0 else None
        else:
            ratios["pe_ratio"] = None
        if equity and equity > 0:
            bvps = equity / shares
            ratios["pb_ratio"] = close_price / bvps if bvps > 0 else None
            ratios["book_value_per_share"] = bvps
        else:
            ratios["pb_ratio"] = None
            ratios["book_value_per_share"] = None
    else:
        ratios["market_cap"] = None
        ratios["pe_ratio"] = None
        ratios["pb_ratio"] = None
        ratios["book_value_per_share"] = None

    # ── Enterprise Value ──────────────────────────────────────────────────
    # EV = Market Cap + Total Debt - Cash
    ev = None
    if market_cap is not None:
        debt = long_term_debt if long_term_debt else 0.0
        cash_val = cash if cash else 0.0
        ev = market_cap + debt - cash_val
        ratios["enterprise_value"] = ev if ev > 0 else None
    else:
        ratios["enterprise_value"] = None

    # ── Earnings Yield (Greenblatt) ───────────────────────────────────────
    # ≈ EBIT / EV (annualized: operating_income * 4 for quarterly data)
    if operating_income and ev and ev > 0:
        annual_ebit = operating_income * 4
        ratios["earnings_yield"] = annual_ebit / ev
    else:
        ratios["earnings_yield"] = None

    # ── Return on Capital (Greenblatt, simplified) ────────────────────────
    # ≈ EBIT / Net Working Capital (PPE not available in pipeline)
    nwc = None
    if current_assets is not None and current_liabilities is not None:
        nwc = current_assets - current_liabilities
    if operating_income and nwc and nwc > 0:
        annual_ebit = operating_income * 4
        ratios["return_on_capital"] = annual_ebit / nwc
    else:
        ratios["return_on_capital"] = None

    # ── Net Current Asset Value (Graham net-nets) ─────────────────────────
    # NCAV = Current Assets - Total Liabilities
    if current_assets is not None and liabilities is not None:
        ncav = current_assets - liabilities
        ratios["ncav"] = ncav
        if shares and shares > 0:
            ratios["ncav_per_share"] = ncav / shares
        else:
            ratios["ncav_per_share"] = None
    else:
        ratios["ncav"] = None
        ratios["ncav_per_share"] = None

    return ratios


def _build_period_structures(market_df, price_map, ticker_map):
    universe = []
    fundamentals = {}
    prices = {}

    market_caps = {}
    for _, row in market_df.iterrows():
        ticker = row.get("ticker")
        if ticker is None or ticker not in ticker_map: continue
        ratios = _compute_derived_ratios(row)
        if ratios.get("market_cap"):
            market_caps[ticker] = ratios["market_cap"]

    sorted_by_mc = sorted(market_caps.items(), key=lambda x: x[1], reverse=True)
    mc_ranks = {ticker: rank + 1 for rank, (ticker, _) in enumerate(sorted_by_mc)}

    for _, row in market_df.iterrows():
        real_ticker = row.get("ticker")
        if real_ticker is None or real_ticker not in ticker_map: continue

        anon = ticker_map[real_ticker]
        ratios = _compute_derived_ratios(row)

        def safe_round(val, decimals=3):
            if val is None: return None
            return round(val, decimals)

        close_val = row.get("close")
        close_rounded = round(float(close_val), 2) if pd.notna(close_val) else None

        universe.append({
            "ticker": anon,
            "market_cap_rank": mc_ranks.get(real_ticker, 999),
            "pe_ratio": safe_round(ratios.get("pe_ratio"), 1),
            "pb_ratio": safe_round(ratios.get("pb_ratio"), 1),
            "market_cap": safe_round(ratios.get("market_cap"), 0),
            "roe": safe_round(ratios.get("roe"), 3),
            "roa": safe_round(ratios.get("roa"), 3),
            "net_income_margin": safe_round(ratios.get("net_income_margin"), 3),
            "debt_to_equity": safe_round(ratios.get("debt_to_equity"), 2),
            "current_ratio": safe_round(ratios.get("current_ratio"), 2),
            "earnings_yield": safe_round(ratios.get("earnings_yield"), 3),
            "revenue_growth_yoy": None,
            "close": close_rounded,
        })

        fund_entry = {}
        raw_fields = {
            "revenue": "revenue", "net_income": "net_income",
            "assets": "assets", "liabilities": "liabilities",
            "cash_equivalents": "cash_equivalents",
            "current_assets": "current_assets", "current_liabilities": "current_liabilities",
            "long_term_debt": "long_term_debt",
            "shares_outstanding": "shares_outstanding",
            "operating_income": "operating_income",
            "net_ppe": "net_ppe"
        }
        for fund_key, col_name in raw_fields.items():
            val = row.get(col_name)
            fund_entry[fund_key] = round(float(val), 2) if pd.notna(val) else None

        for ohlcv_col in ["open", "high", "low", "close", "volume"]:
            val = row.get(ohlcv_col)
            if ohlcv_col == "volume":
                fund_entry[ohlcv_col] = int(float(val)) if pd.notna(val) else None
            else:
                fund_entry[ohlcv_col] = round(float(val), 2) if pd.notna(val) else None

        for ratio_name, ratio_val in ratios.items():
            fund_entry[ratio_name] = safe_round(ratio_val, 4)

        fundamentals[anon] = fund_entry

        if real_ticker in price_map and price_map[real_ticker] is not None:
            prices[anon] = round(price_map[real_ticker], 2)

    universe.sort(key=lambda x: x.get("market_cap_rank", 999))
    return universe, fundamentals, prices


# ══════════════════════════════════════════════════════════════════════════════
# Revenue Growth (YoY)
# ══════════════════════════════════════════════════════════════════════════════

def _enrich_revenue_growth(fundamentals_by_period, periods):
    for i, period in enumerate(periods):
        if i < 4: continue
        prior_period = periods[i - 4]
        current_funds = fundamentals_by_period.get(period, {})
        prior_funds = fundamentals_by_period.get(prior_period, {})
        for anon_ticker, current_data in current_funds.items():
            prior_data = prior_funds.get(anon_ticker)
            if prior_data is None: continue
            curr_rev = current_data.get("revenue")
            prior_rev = prior_data.get("revenue")
            if curr_rev and prior_rev and prior_rev > 0:
                current_data["revenue_growth_yoy"] = round((curr_rev - prior_rev) / prior_rev, 4)


# ══════════════════════════════════════════════════════════════════════════════
# Dataset Persistence (Save/Load)
# ══════════════════════════════════════════════════════════════════════════════

def save_dataset(
    periods, market_universe_by_period, fundamentals_by_period, prices_by_period,
    ticker_map, final_val_prices, save_path: str
):
    """
    Save prepared backtest dataset to a JSON file for later reuse.

    Parameters
    ----------
    periods : list[str]
        Period labels (e.g., ["Period_001", "Period_002", ...])
    market_universe_by_period : dict
        Market universe data keyed by period
    fundamentals_by_period : dict
        Fundamental data keyed by period
    prices_by_period : dict
        Price data keyed by period
    ticker_map : dict
        Mapping of real to anonymized tickers
    final_val_prices : dict
        Final valuation prices at end date
    save_path : str
        File path to save the dataset (typically .json)
    """
    import json
    import os

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    # Create reverse mapping (anonymized → real) for reference
    reverse_ticker_map = {v: k for k, v in ticker_map.items()}
    real_tickers = list(ticker_map.keys())

    payload = {
        "periods": periods,
        "market_universe_by_period": market_universe_by_period,
        "fundamentals_by_period": fundamentals_by_period,
        "prices_by_period": prices_by_period,
        "ticker_map": ticker_map,
        "reverse_ticker_map": reverse_ticker_map,
        "real_tickers": real_tickers,
        "final_val_prices": final_val_prices,
    }

    with open(save_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    print(f"  Dataset saved to {save_path}")


def load_dataset(load_path: str) -> tuple:
    """
    Load a pre-recorded backtest dataset from a JSON file.
    
    Returns the exact same 6-tuple that prepare_backtest_data() would return:
    (periods, market_universe_by_period, fundamentals_by_period,
     prices_by_period, ticker_map, final_val_prices)

    Parameters
    ----------
    load_path : str
        File path to load the dataset from

    Returns
    -------
    tuple
        (periods, market_universe_by_period, fundamentals_by_period,
         prices_by_period, ticker_map, final_val_prices)

    Raises
    ------
    FileNotFoundError
        If the load_path does not exist
    """
    import json

    if not Path(load_path).exists():
        raise FileNotFoundError(f"Dataset file not found: {load_path}")

    with open(load_path, "r") as f:
        payload = json.load(f)

    print(f"  Dataset loaded from {load_path}")

    # Return in same order and structure as prepare_backtest_data()
    periods = payload["periods"]
    market_universe_by_period = payload["market_universe_by_period"]
    fundamentals_by_period = payload["fundamentals_by_period"]
    prices_by_period = payload["prices_by_period"]
    ticker_map = payload["ticker_map"]
    final_val_prices = payload["final_val_prices"]

    return periods, market_universe_by_period, fundamentals_by_period, prices_by_period, ticker_map, final_val_prices


# ══════════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def prepare_backtest_data(
    start_date="2014-01-01", end_date="2023-12-31", frequency="quarterly",
    n_stocks=None, wrds_window_days=10,
    load_path: Optional[str] = None, save_path: Optional[str] = None,
):
    """
    Prepare backtest data by loading from WRDS, using synthetic data, or loading from a file.

    Parameters
    ----------
    start_date : str
        Start date (YYYY-MM-DD, default: 2014-01-01)
    end_date : str
        End date (YYYY-MM-DD, default: 2023-12-31)
    frequency : str
        Rebalancing frequency: "quarterly", "monthly", "semi-annual", "annual" (default: quarterly)
    n_stocks : int, optional
        Limit to top N stocks by market cap (default: None = all)
    wrds_window_days : int
        Data fetch window in days for WRDS queries (default: 10)
    load_path : str, optional
        Path to a pre-recorded dataset .json file. If provided, loads from this instead of fetching.
    save_path : str, optional
        Path to save the dataset .json file after fetching (ignored if load_path is used)

    Returns
    -------
    tuple
        (periods, market_universe_by_period, fundamentals_by_period,
         prices_by_period, ticker_map, final_val_prices)
    """
    # Check if loading from a pre-recorded dataset
    if load_path:
        print(f"  Loading pre-recorded dataset from {load_path}...")
        result = load_dataset(load_path)
        return result

    rebalance_dates = generate_rebalance_dates(start_date, end_date, frequency)
    periods = [f"Period_{i+1:03d}" for i in range(len(rebalance_dates))]

    print(f"  Backtest: {start_date} -> {end_date} ({frequency})")
    print(f"  {len(periods)} trading periods generated")

    last_rebalance = pd.Timestamp(rebalance_dates[-1]) if rebalance_dates else None
    end_ts = pd.Timestamp(end_date)
    needs_final_valuation = last_rebalance is not None and end_ts > last_rebalance
    if needs_final_valuation:
        print(f"  Final valuation at {end_date} ({(end_ts - last_rebalance).days}d after last trade)")

    print("  Connecting to WRDS...")
    market_universe_by_period = {}; fundamentals_by_period = {}; prices_by_period = {}
    all_tickers = set(); period_raw_data = {}

    with WRDSDATA() as wrds_conn:
        print("  [Pass 1] Fetching data for each period...")
        for i, (period, date) in enumerate(zip(periods, rebalance_dates)):
            print(f"    {period} ({date})...", end=" ", flush=True)
            market_df, price_map, ticker_list = _fetch_period_data(wrds_conn, date, window_days=wrds_window_days)
            if market_df.empty: print("no data"); continue
            period_raw_data[period] = (market_df, price_map, ticker_list)
            all_tickers.update(ticker_list)
            print(f"{len(ticker_list)} stocks")

        ticker_map = build_ticker_map(sorted(all_tickers))

        final_val_prices = {}
        if needs_final_valuation:
            print(f"    Final valuation ({end_date})...", end=" ", flush=True)
            final_val_prices = _fetch_prices_only(wrds_conn, end_date, ticker_map, window_days=wrds_window_days)
            print(f"{len(final_val_prices)} prices")

    print(f"  Ticker map: {len(ticker_map)} unique tickers anonymized")
    print("  [Pass 2] Building period data structures...")

    for period in periods:
        if period not in period_raw_data:
            market_universe_by_period[period] = []; fundamentals_by_period[period] = {}; prices_by_period[period] = {}
            continue
        market_df, price_map, _ = period_raw_data[period]
        universe, funds, prices = _build_period_structures(market_df, price_map, ticker_map)
        if n_stocks is not None:
            top_tickers = set(entry["ticker"] for entry in universe[:n_stocks])
            universe = [u for u in universe if u["ticker"] in top_tickers]
            funds = {t: f for t, f in funds.items() if t in top_tickers}
            prices = {t: p for t, p in prices.items() if t in top_tickers}
        market_universe_by_period[period] = universe
        fundamentals_by_period[period] = funds
        prices_by_period[period] = prices

    if n_stocks is not None and final_val_prices:
        last_period_tickers = set(prices_by_period.get(periods[-1], {}).keys())
        final_val_prices = {t: p for t, p in final_val_prices.items() if t in last_period_tickers}

    _enrich_revenue_growth(fundamentals_by_period, periods)
    for period in periods:
        for entry in market_universe_by_period.get(period, []):
            anon = entry["ticker"]
            fund_data = fundamentals_by_period.get(period, {}).get(anon, {})
            if fund_data.get("revenue_growth_yoy") is not None:
                entry["revenue_growth_yoy"] = fund_data["revenue_growth_yoy"]

    n_with_data = sum(1 for p in periods if prices_by_period.get(p))
    print(f"  Done: {n_with_data}/{len(periods)} periods with data")
    if final_val_prices:
        print(f"  Final valuation prices: {len(final_val_prices)} tickers at {end_date}")

    # Save dataset if save_path is provided
    if save_path:
        save_dataset(periods, market_universe_by_period, fundamentals_by_period,
                     prices_by_period, ticker_map, final_val_prices, save_path)

    return periods, market_universe_by_period, fundamentals_by_period, prices_by_period, ticker_map, final_val_prices


# ══════════════════════════════════════════════════════════════════════════════
# Entry Point to save data
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare backtest data via WRDS, synthetic, or loading.")
    parser.add_argument("--start", default="2014-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2023-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--frequency", default="quarterly", choices=["quarterly", "monthly", "semi-annual", "annual"],
                        help="Rebalancing frequency")
    parser.add_argument("--n-stocks", type=int, default=None, help="Limit to top N stocks by market cap")
    parser.add_argument("--save_path", type=str, default="private_results/data/backtest_data.json",
                        help="Save dataset to this path")
    
    args = parser.parse_args()
        
    print("\n" + "="*70)
    print("DATA LOADER — prepare_backtest_data()")
    print("="*70)
    
    periods, universe, funds, prices, ticker_map, final_vals = prepare_backtest_data(
        start_date=args.start,
        end_date=args.end,
        frequency=args.frequency,
        n_stocks=args.n_stocks,
        load_path=None,
        save_path=args.save_path,
    )
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"  Periods:         {len(periods)}")
    print(f"  Tickers:         {len(ticker_map)}")
    first_period = periods[0] if periods else None
    if first_period:
        n_stocks_first = len(universe.get(first_period, []))
        print(f"  Stocks (P1):     {n_stocks_first}")
    print(f"  Final vals:      {len(final_vals)} prices")
    print("="*70 + "\n")