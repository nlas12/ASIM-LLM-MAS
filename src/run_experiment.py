"""
Master Experiment Runner — Phase 1 Refactor
run_experiment.py
=============================================
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from data_loader import prepare_backtest_data
from compute_benchmarks import build_benchmark_series, fetch_msci_world_index
import pipelines.single_agent_pipeline as single_agent_pipeline
from pipelines.single_agent_pipeline import run_backtest, INITIAL_CAPITAL
from pipelines.multi_agent_pipeline import run_multiagent_backtest
from portfolio import compute_metrics
from experiment_logger import ExperimentLogger, set_logger
from personas import PERSONAS

PERSONA_NAMES = ["buffett", "cathie_wood", "ray_dalio", "ben_graham", "joel_greenblatt"]
COORDINATION_MECHANISMS = ["majority_vote", "average_size", "llm_manager"]
N_RUNS = 1


def main():
    parser = argparse.ArgumentParser(
        description="ASIM WS25/26 — LLM-Based Investor Agent Backtest"
    )
    parser.add_argument("--start", default="2014-01-01")
    parser.add_argument("--end", default="2014-12-31")
    parser.add_argument("--frequency", default="quarterly",
                        choices=["quarterly", "monthly", "semi-annual", "annual"])
    parser.add_argument("--n_runs", type=int, default=N_RUNS)
    parser.add_argument("--n_stocks", type=int, default=None,
                        help="Limit to top N stocks by market cap (default: all NASDAQ-100)")
    parser.add_argument("--output", default="results/final/experiment_results.json")
    parser.add_argument("--personas_only", action="store_true",
                        help="Only run single-agent baselines (skip multi-agent)")
    parser.add_argument("--multi_only", action="store_true",
                        help="Only run multi-agent coordination (skip single-agent baselines)")
    parser.add_argument("--personas", nargs="+", default=None,
                        help="Subset of personas to run (e.g. --personas buffett ray_dalio)")
    parser.add_argument("--coordination", nargs="+", default=None,
                        help="Subset of coordination mechanisms to test")
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="Global LLM temperature for screening/analysis (default: 0.5). "
                             "Decision nodes always use 0.1 for deterministic trade orders.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers for persona/coordination runs (default: 1 = sequential)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    persona_names = args.personas or PERSONA_NAMES
    coord_mechanisms = args.coordination or COORDINATION_MECHANISMS

    invalid = [p for p in persona_names if p not in PERSONAS]
    if invalid:
        print(f"ERROR: Unknown persona(s): {invalid}")
        print(f"Available: {list(PERSONAS.keys())}")
        return

    if args.personas_only and args.multi_only:
        print("ERROR: --personas_only and --multi_only are mutually exclusive.")
        return

    single_agent_pipeline.LLM_TEMPERATURE = args.temperature

    log_path = args.output.replace(".json", "_log.json") if args.output else "results/final/experiment_log.json"
    exp_logger = ExperimentLogger(log_path)
    set_logger(exp_logger)

    print("=" * 70)
    print("ASIM WS25/26 — LLM-Based Investor Agent Backtest Experiment")
    print(f"Start: {args.start} | End: {args.end} | Freq: {args.frequency} | Runs: {args.n_runs}")
    print(f"Temperature: {args.temperature} (decisions: {single_agent_pipeline.DECISION_TEMPERATURE})")
    print(f"Workers: {args.workers}")
    print(f"Personas: {', '.join(persona_names)}")
    if not args.personas_only:
        print(f"Coordination: {', '.join(coord_mechanisms)}")
    print("=" * 70)

    # ── 1. Data Preparation ───────────────────────────────────────────────
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

    # Build market caps from fundamentals (shares_outstanding × price)
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
        "cap_weighted": build_benchmark_series(prices, periods, strategy="cap_weighted", final_valuation_prices=final_valuation_prices or None, market_caps_by_period=market_caps_by_period),
        "equal_weight": build_benchmark_series(prices, periods, strategy="equal_weight", final_valuation_prices=final_valuation_prices or None),
        "msci_world": fetch_msci_world_index(start_date=args.start, end_date=args.end, frequency="quarterly"),
    }

    all_results = {
        "experiment_meta": {
            "start_date": args.start,
            "end_date": args.end,
            "n_periods": len(periods),
            "has_final_valuation": bool(final_valuation_prices),
            "n_runs": args.n_runs,
            "initial_capital": INITIAL_CAPITAL,
            "personas": persona_names,
            "coordination_mechanisms": coord_mechanisms if not args.personas_only else [],
            "temperature": args.temperature,
            "decision_temperature": single_agent_pipeline.DECISION_TEMPERATURE,
            "timestamp": datetime.now().isoformat(),
        },
        "benchmarks": {},
        "single_agent": {},
        "multi_agent": {},
    }

    for bench_name, bench_series in benchmarks.items():
        period_values = [{"period": k, "portfolio_value": v} for k, v in bench_series.items()]
        all_results["benchmarks"][bench_name] = {
            "period_values": period_values,
            "metrics": compute_metrics(period_values, INITIAL_CAPITAL),
        }
    print(f"      Benchmarks computed: {list(benchmarks.keys())}")

    # Checkpoint helper — saves partial results after each completed run
    def _save_checkpoint():
        exp_logger.save()
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # ── 2. Single-Agent Baselines ─────────────────────────────────────────
    if args.multi_only:
        print("\n[2/3] Skipping single-agent baselines (--multi_only flag set).")
    else:
        n_single = len(persona_names) * args.n_runs
        print(f"\n[2/3] Running single-agent baselines ({len(persona_names)} personas × {args.n_runs} runs = {n_single} total)...")

        def _run_single_persona(persona):
            exp_logger.set_context(pipeline="single_agent", persona=persona)
            runs = run_backtest(
                persona_name=persona,
                periods=periods,
                market_universe_by_period=universe,
                fundamentals_by_period=funds,
                prices_by_period=prices,
                n_runs=args.n_runs,
                reverse_ticker_map=reverse_ticker_map,
                final_valuation_prices=final_valuation_prices or None,
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

        if args.workers > 1:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(_run_single_persona, p): p for p in persona_names}
                for future in as_completed(futures):
                    persona, result = future.result()
                    all_results["single_agent"][persona] = result
                    _save_checkpoint()
        else:
            for persona in persona_names:
                print(f"\n  Persona: {persona.upper()}")
                _, result = _run_single_persona(persona)
                all_results["single_agent"][persona] = result
                _save_checkpoint()

    # ── 3. Multi-Agent ────────────────────────────────────────────────────
    if args.personas_only:
        print("\nSkipping multi-agent (--personas_only flag set).")
    else:
        n_multi = len(coord_mechanisms) * args.n_runs
        print(f"\n[3/3] Running multi-agent ({len(coord_mechanisms)} mechanisms × {args.n_runs} runs = {n_multi} total)...")

        def _run_multi_coord(coord):
            exp_logger.set_context(pipeline="multi_agent")
            runs = run_multiagent_backtest(
                persona_names=persona_names,
                periods=periods,
                market_universe_by_period=universe,
                fundamentals_by_period=funds,
                prices_by_period=prices,
                coordination=coord,
                n_runs=args.n_runs,
                reverse_ticker_map=reverse_ticker_map,
                final_valuation_prices=final_valuation_prices or None,
                workers=args.workers,
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

        if args.workers > 1:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(_run_multi_coord, c): c for c in coord_mechanisms}
                for future in as_completed(futures):
                    coord, result = future.result()
                    all_results["multi_agent"][coord] = result
                    _save_checkpoint()
        else:
            for coord in coord_mechanisms:
                print(f"\n  Coordination: {coord.upper()}")
                _, result = _run_multi_coord(coord)
                all_results["multi_agent"][coord] = result
                _save_checkpoint()

    # ── Final save ────────────────────────────────────────────────────────
    exp_logger.print_summary()
    exp_logger.save()

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n✓ Results saved to {args.output}")

    print_summary_table(all_results)


def _mean_metrics(metrics_list):
    if not metrics_list: return {}
    keys = [k for k in metrics_list[0] if isinstance(metrics_list[0][k], (int, float))]
    result = {}
    for k in keys:
        vals = [m[k] for m in metrics_list if k in m and isinstance(m[k], (int, float))]
        result[k] = round(sum(vals) / len(vals), 3) if vals else None
    return result


def print_summary_table(results):
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"{'Condition':<30} {'Ann. Return':>12} {'Ann. Vol':>10} {'Sharpe':>8} {'Max DD':>10}")
    print("-" * 80)

    def print_row(name, metrics):
        ret = metrics.get("annualized_return_pct", "N/A")
        vol = metrics.get("annualized_volatility_pct", "N/A")
        sharpe = metrics.get("sharpe_ratio", "N/A")
        dd = metrics.get("max_drawdown_pct", "N/A")
        print(f"{name:<30} {str(ret) + ' %':>12} {str(vol) + ' %':>10} {str(sharpe):>8} {str(dd) + ' %':>10}")

    for name, data in results.get("benchmarks", {}).items():
        print_row(f"[Benchmark] {name}", data.get("metrics", {}))
    print()
    for persona, data in results.get("single_agent", {}).items():
        print_row(f"[Single] {persona}", data.get("metrics_mean", {}))
    print()
    for coord, data in results.get("multi_agent", {}).items():
        print_row(f"[Multi] {coord}", data.get("metrics_mean", {}))
    print("=" * 80)


if __name__ == "__main__":
    main()