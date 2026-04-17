"""
Master Experiment Runner — Phase 1 Refactor
run_experiment.py
=============================================
"""

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from data_loader import prepare_backtest_data
from compute_benchmarks import build_benchmark_series, fetch_msci_world_index
import pipelines.single_agent_pipeline as single_agent_pipeline
from pipelines.single_agent_pipeline import run_backtest
from pipelines.multi_agent_pipeline import run_multiagent_backtest
from personas import PERSONAS

def main():
    # ── Setup ─────────────────────────────────────────────────────────────
    args = _setup_arguments()
    os.makedirs(args.output, exist_ok=True)
    
    persona_names, coord_mechanisms = _validate_and_setup(args)
    if persona_names is None:
        return
    
    _print_startup_info(args, persona_names, coord_mechanisms)

    # ── Data Preparation ──────────────────────────────────────────────────
    (periods, universe, funds, prices, reverse_ticker_map, 
     final_valuation_prices, benchmarks) = _prepare_data(args)

    # ── Run Experiments ───────────────────────────────────────────────────
    if args.mode in ["single", "both"]:
        _run_single_agent_experiments(args, persona_names, periods, universe, funds, prices,
                                      reverse_ticker_map, final_valuation_prices)
    
    if args.mode in ["multi", "both"]:
        _run_multi_agent_experiments(args, persona_names, coord_mechanisms, periods, universe, funds,
                                     prices, final_valuation_prices)

    # ── Done ───────────────────────────────────────────────────────────────
    print(f"\n✓ Experiment folders saved to: {args.output}")

def _setup_arguments():
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description="ASIM WS25/26 — LLM-Based Investor Agent Backtest"
    )
    parser.add_argument("--start", default="2014-01-01")
    parser.add_argument("--end", default="2014-12-31")
    parser.add_argument("--frequency", default="quarterly",
                        choices=["quarterly", "monthly", "semi-annual", "annual"])
    parser.add_argument("--n_runs", type=int, default=1,
                        help="Number of runs per persona/coordination (default: 1)")
    parser.add_argument("--n_stocks", type=int, default=None,
                        help="Limit to top N stocks by market cap (default: all NASDAQ-100)")
    parser.add_argument("--output", default="results",
                        help="Root folder where experiment folders will be created")
    parser.add_argument("--mode", choices=["single", "multi", "both"], default="multi",
                        help="Which experiments to run: single-agent, multi-agent, or both (default: multi)")
    parser.add_argument("--personas", nargs="+", default=None,
                        help="Subset of personas to run (default: all)")
    parser.add_argument("--coordination", nargs="+", default=None,
                        help="Subset of coordination mechanisms to test (default: all)")
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="Global LLM temperature for screening/analysis (default: 0.5). "
                             "Decision nodes always use 0.1 for deterministic trade orders.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers for persona/coordination runs (default: 1 = sequential)")
    return parser.parse_args()

def _validate_and_setup(args):
    """Validate arguments and set configuration."""
    persona_names = (args.personas 
                     or ["buffett", "cathie_wood", "ray_dalio", "ben_graham", "joel_greenblatt"])
    coord_mechanisms = (args.coordination 
                        or ["majority_vote", "average_size", "llm_manager"])
    
    invalid = [p for p in persona_names if p not in PERSONAS]
    if invalid:
        print(f"ERROR: Unknown persona(s): {invalid}")
        print(f"Available: {list(PERSONAS.keys())}")
        return None, None

    single_agent_pipeline.LLM_TEMPERATURE = args.temperature
    return persona_names, coord_mechanisms

def _print_startup_info(args, persona_names, coord_mechanisms):
    """Print experiment configuration."""
    print("=" * 70)
    print("ASIM WS25/26 — LLM-Based Investor Agent Backtest Experiment")
    print(f"Start: {args.start} | End: {args.end} | Freq: {args.frequency} | Runs: {args.n_runs}")
    print(f"Mode: {args.mode.upper()}")
    print(f"Temperature: {args.temperature} (decisions: {single_agent_pipeline.DECISION_TEMPERATURE})")
    print(f"Workers: {args.workers}")
    print(f"Personas: {', '.join(persona_names)}")
    print(f"Coordination: {', '.join(coord_mechanisms)}")
    print("=" * 70)

def _prepare_data(args):
    """Load market data and compute benchmarks."""
    print("\n[1/3] Preparing data...")
    (periods, universe, funds, prices,
     ticker_map, final_valuation_prices) = prepare_backtest_data(
        start_date=args.start,
        end_date=args.end,
        frequency=args.frequency,
        n_stocks=args.n_stocks,
    )
    reverse_ticker_map = {v: k for k, v in ticker_map.items()}
    print(f"      {len(periods)} periods | {len(ticker_map)} stocks")
    if final_valuation_prices:
        print(f"      Final valuation at {args.end} ({len(final_valuation_prices)} prices)")

    # Build market caps from fundamentals
    market_caps_by_period = {}
    for period in periods:
        fund = funds.get(period, {})
        pp = prices.get(period, {})
        caps = {}
        for ticker in pp:
            f = fund.get(ticker, {})
            shares = f.get("shares_outstanding")
            price = pp.get(ticker)
            if shares and price and shares > 0 and price > 0:
                caps[ticker] = shares * price
        market_caps_by_period[period] = caps

    benchmarks = {
        "cap_weighted": build_benchmark_series(prices, periods, strategy="cap_weighted", 
                                               final_valuation_prices=final_valuation_prices or None, 
                                               market_caps_by_period=market_caps_by_period),
        "equal_weight": build_benchmark_series(prices, periods, strategy="equal_weight", 
                                               final_valuation_prices=final_valuation_prices or None),
        "msci_world": fetch_msci_world_index(start_date=args.start, end_date=args.end, frequency="quarterly"),
    }
    print(f"      Benchmarks computed: {list(benchmarks.keys())}")
    
    return periods, universe, funds, prices, reverse_ticker_map, final_valuation_prices, benchmarks



def _run_tasks_parallel_or_sequential(task_items, task_fn, workers):
    """Execute tasks either in parallel or sequentially."""
    results = {}
    
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(task_fn, item): item for item in task_items}
            for future in as_completed(futures):
                key, result = future.result()
                results[key] = result
    else:
        for item in task_items:
            key, result = task_fn(item)
            results[key] = result
    
    return results

def _run_single_agent_experiments(args, persona_names, periods, universe, funds, prices,
                                  reverse_ticker_map, final_valuation_prices):
    """Run single-agent baseline experiments."""
    single_experiment_folder = os.path.join(args.output, _get_experiment_subfolder("single_agent"))
    n_single = len(persona_names) * args.n_runs
    print(f"\n[2/3] Running single-agent baselines ({len(persona_names)} personas × {args.n_runs} runs = {n_single} total)...")
    print(f"      Saving to: {single_experiment_folder}")

    def _run_single_persona(persona):
        runs = run_backtest(
            persona_name=persona,
            periods=periods,
            market_universe_by_period=universe,
            fundamentals_by_period=funds,
            prices_by_period=prices,
            n_runs=args.n_runs,
            reverse_ticker_map=reverse_ticker_map,
            final_valuation_prices=final_valuation_prices or None,
            experiment_root_dir=single_experiment_folder,
        )
        all_run_metrics = [r["performance"] for r in runs]
        result = {
            "runs": runs,
            "metrics_per_run": all_run_metrics,
            "metrics_mean": _mean_metrics(all_run_metrics),
        }
        mean_ret = result["metrics_mean"].get("annualized_return_pct", "?")
        print(f"  Done [{persona.upper()}]. Mean annualized return: {mean_ret}%")
        return persona, result

    _run_tasks_parallel_or_sequential(persona_names, _run_single_persona, args.workers)

def _run_multi_agent_experiments(args, persona_names, coord_mechanisms, periods, universe, funds, prices,
                                 final_valuation_prices):
    """Run multi-agent coordination experiments."""
    multi_experiment_folder = os.path.join(args.output, _get_experiment_subfolder("multi_agent"))
    n_multi = len(coord_mechanisms) * args.n_runs
    print(f"\n[3/3] Running multi-agent ({len(coord_mechanisms)} mechanisms × {args.n_runs} runs = {n_multi} total)...")
    print(f"      Saving to: {multi_experiment_folder}")

    def _run_multi_coord(coord):
        runs = run_multiagent_backtest(
            persona_names=persona_names,
            periods=periods,
            market_universe_by_period=universe,
            fundamentals_by_period=funds,
            prices_by_period=prices,
            coordination=coord,
            n_runs=args.n_runs,
            final_valuation_prices=final_valuation_prices or None,
            workers=args.workers,
            experiment_root_dir=multi_experiment_folder,
        )
        all_run_metrics = [r["performance"] for r in runs]
        result = {
            "runs": runs,
            "metrics_per_run": all_run_metrics,
            "metrics_mean": _mean_metrics(all_run_metrics),
        }
        mean_ret = result["metrics_mean"].get("annualized_return_pct", "?")
        print(f"  Done [{coord.upper()}]. Mean annualized return: {mean_ret}%")
        return coord, result

    _run_tasks_parallel_or_sequential(coord_mechanisms, _run_multi_coord, args.workers)

def _get_experiment_subfolder(experiment_type: str) -> str:
    """Generate timestamped experiment subfolder name."""
    now = datetime.now()
    timestamp = f"{now.strftime('%m%d')}.{now.strftime('%H%M')}.{int(time.time() % 60):02d}"
    return f"{experiment_type}_{timestamp}"


def _mean_metrics(metrics_list):
    if not metrics_list: return {}
    keys = [k for k in metrics_list[0] if isinstance(metrics_list[0][k], (int, float))]
    result = {}
    for k in keys:
        vals = [m[k] for m in metrics_list if k in m and isinstance(m[k], (int, float))]
        result[k] = round(sum(vals) / len(vals), 3) if vals else None
    return result

if __name__ == "__main__":
    main()