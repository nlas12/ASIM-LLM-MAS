# ASIM-LLM-MAS

Adaptive System for Investment Management with Large Language Models using Multi-Agent Systems.

## Overview

This project implements a **multi-agent investment simulation system** where LLM-based investor personas (Warren Buffett, Benjamin Graham, Cathie Wood, Joel Greenblatt, Ray Dalio) collaborate to make portfolio decisions. The system uses a two-pass LangGraph pipeline with peer-informed reasoning and coordination mechanisms based on real financial data from WRDS (Wharton Research Data Services).

## Architecture

### Single-Agent Pipeline (3-step)
Each persona independently processes market data through:

```
screening (Step 1) → analysis (Step 2) → decision (Step 3) → execution
     ↓                  ↓                  ↓
 Select 8-12      Deep financial      Trade orders
 candidates       analysis + signals   (BUY/SELL/HOLD)
```

### Multi-Agent Pipeline (8-step with coordination)
Multiple personas collaborate with peer awareness:

```
PASS 1 (Parallel)        PASS 2 (Parallel)       Coordination
  Steps 1-3   ────→  Step 4: Summary  ────→  Step 5-6: Reanalysis  ────→  Step 7-8
                        Build                  with peers               Aggregate
                                                                      & Execute
```

See [agent_architecture.md](agent_architecture.md) for detailed workflow documentation.

## Key Features

| Feature                      | Description                                              |
| ---------------------------- | -------------------------------------------------------- |
| **5 Investor Personas**      | Buffett, Graham, Cathie Wood, Dalio, Greenblatt          |
| **Multi-Agent Coordination** | Majority Vote, Average Size, LLM Manager                 |
| **2-Pass Pipeline**          | Independent decisions + peer-informed revision           |
| **Memory System**            | Trade history, position tracking, return statistics      |
| **Benchmarking**             | MSCI World Index comparisons in `compute_benchmarks.py`  |
| **Comprehensive Logging**    | All LLM calls logged with system/human prompts & outputs |

## Key Components

| Module                  | Description                                          |
| ----------------------- | ---------------------------------------------------- |
| `single_agent_pipeline` | 3-node LangGraph for individual investor agents      |
| `multi_agent_pipeline`  | 8-step pipeline with coordination mechanisms         |
| `portfolio.py`          | Trade execution, position tracking, P&L calculation  |
| `personas.py`           | 5 investor personas with system prompts              |
| `data_loader.py`        | WRDS data preparation and filtering                  |
| `experiment_logger.py`  | LLM call logging and audit trail                     |
| `compute_benchmarks.py` | MSCI World Index fetching and benchmark construction |

## Project Structure

```
├── README.md                          # This file
├── agent_architecture.md              # Detailed pipeline documentation
├── environment.yml                    # Conda environment
├── benchmarks.json                    # Pre-computed benchmark data
├── 01_Result_Analysis.ipynb           # Jupyter notebook for results visualization
│
├── data/
│   ├── __init__.py
│   └── wrds_data.py                   # WRDS connection & queries
│
├── src/
│   ├── run_experiment.py              # Main entry point (CLI)
│   ├── single_agent_pipeline.py       # 3-step single-agent LangGraph
│   ├── multi_agent_pipeline.py        # 8-step multi-agent coordination
│   ├── personas.py                    # Investor personas & prompts
│   ├── portfolio.py                   # Trade execution & ledger
│   ├── data_loader.py                 # Prepare WRDS data for backtests
│   ├── compute_benchmarks.py          # MSCI World Index benchmarks
│   └── experiment_logger.py           # LLM call logging
│
└── results/
    ├── portfolios_single/             # Single-agent results (run1-10)
    └── portfolios_multi/              # Multi-agent results (run1-10)
```

## Requirements

- Python 3.11+
- WRDS account with credentials in `.pgpass` file (see [WRDS documentation](https://wrds-www.wharton.upenn.edu/))
- Google API key for Gemini LLM

## Environment Setup

```bash
# Create conda environment
conda env create -f environment.yml
conda activate asim-llm-mas

# Configure credentials
# 1. Set up WRDS in ~/.pgpass (passwordless access)
# 2. Create .env file in project root:
#    WRDS_USERNAME=your_wrds_username
#    GOOGLE_API_KEY=your_google_api_key
```

## Usage

### Basic Single-Agent Run

Run a single persona backtest:

```bash
python src/run_experiment.py \
  --personas buffett \
  --start 2014-01-01 \
  --end 2023-12-31 \
  --n_runs 3
```

### Multi-Agent Coordination Run

Compare all 5 personas with coordination mechanisms:

```bash
python src/run_experiment.py \
  --personas buffett cathie_wood ray_dalio ben_graham joel_greenblatt \
  --start 2014-01-01 \
  --end 2023-12-31 \
  --frequency quarterly \
  --n_runs 3
```

### Command-Line Options

```
--start DATE                          Start date (YYYY-MM-DD, default: 2014-01-01)
--end DATE                            End date (YYYY-MM-DD, default: 2023-12-31)
--frequency {quarterly|monthly|...}   Rebalancing frequency (default: quarterly)
--n_runs N                            Number of simulation runs (default: 3)
--n_stocks N                          Limit to top N stocks by market cap (default: all)
--output FILE                         Results output path
--personas_only                       Skip multi-agent runs
--multi_only                          Skip single-agent baselines
--personas P1 P2 ...                  Specific personas to run
--coordination C1 C2 ...              Coordination mechanisms to test
--temperature T                       LLM temperature for screening/analysis (default: 0.5)
--workers N                           Parallel agents per pass (default: 1)
```

### Available Personas

- `buffett` — Warren Buffett (value investing, ROE-focused)
- `ben_graham` — Benjamin Graham (margin of safety, intrinsic value)
- `cathie_wood` — Cathie Wood (disruptive innovation, growth)
- `ray_dalio` — Ray Dalio (macro diversification)
- `joel_greenblatt` — Joel Greenblatt (ROIC, earnings yield)

### Available Coordination Mechanisms

- `majority_vote` — BUY if >50% agents agree
- `average_size` — Aggregate average position sizes
- `llm_manager` — LLM synthesizes all proposals into final orders

## Results Analysis

After running experiments, explore results:

```bash
jupyter notebook 01_Result_Analysis.ipynb
```

Results are saved to:
- `results/portfolios_single/` — Individual persona backtests
- `results/portfolios_multi/` — Multi-agent coordination results

## Publications & Citation

This system was developed for:
- **Course:** Advanced Seminar Information Management (WS 25/26)
- **Institution:** University of Cologne

## License

[LICENSE](LICENSE) — See LICENSE file for details.
