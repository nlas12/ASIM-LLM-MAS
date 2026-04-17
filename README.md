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
├── experiment_folder_structure.md     # Experiment output folder format
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
│   ├── personas.py                    # Investor personas & prompts
│   ├── portfolio.py                   # Trade execution & ledger
│   ├── data_loader.py                 # Prepare WRDS data for backtests
│   ├── compute_benchmarks.py          # MSCI World Index benchmarks
│   ├── experiment_logger.py           # LLM call logging & audit trail
│   │
│   └── pipelines/
│       ├── single_agent_pipeline.py   # 3-step single-agent LangGraph
│       ├── multi_agent_pipeline.py    # 8-step multi-agent coordination
│       └── pipeline_utils.py          # Shared models, memory, LLM utilities
│
└── results/
    └── experiments/                   # Timestamped experiment folders
        ├── single_agent_MMDD.HHMM.SS/ # Single-agent experiment
        │   ├── run1/
        │   ├── run2/
        │   └── run3/
        └── multi_agent_MMDD.HHMM.SS/  # Multi-agent experiment
            ├── run1/
            ├── run2/
            └── run3/
```

## Requirements

- Python 3.11+
- WRDS account with credentials in `.pgpass` file (see [WRDS documentation](https://wrds-www.wharton.upenn.edu/))
- Google API key for Gemini LLM

## Environment Setup

```bash
# Create conda environment
conda env create -f environment.yml
conda activate mas_env
```

# Configure credentials
1. Set up WRDS in ~/.pgpass (passwordless access)
2. Create .env file in project root:
```bash
WRDS_USERNAME=your_wrds_username
GOOGLE_API_KEY=your_google_api_key
```


## Usage

### Default: Multi-Agent Run (Multi-Agent Coordination Only)

```bash
python src/run_experiment.py \
  --start 2014-01-01 \
  --end 2023-12-31 \
  --frequency quarterly \
  --n_runs 3
```

### Single-Agent Run (All 5 Personas as Independent Baselines)

```bash
python src/run_experiment.py \
  --mode single \
  --start 2014-01-01 \
  --end 2023-12-31 \
  --n_runs 3
```

### Both Single and Multi-Agent Experiments

```bash
python src/run_experiment.py \
  --mode both \
  --start 2014-01-01 \
  --end 2023-12-31 \
  --n_runs 3
```

### Subset of Personas and Coordination Mechanisms

```bash
python src/run_experiment.py \
  --mode both \
  --personas buffett cathie_wood ray_dalio \
  --coordination majority_vote average_size \
  --n_runs 5
```

### Command-Line Options

```
--start DATE                          Start date (YYYY-MM-DD, default: 2014-01-01)
--end DATE                            End date (YYYY-MM-DD, default: 2023-12-31)
--frequency {quarterly|monthly|...}   Rebalancing frequency (default: quarterly)
--n_runs N                            Number of simulation runs (default: 1)
--n_stocks N                          Limit to top N stocks by market cap (default: all)
--output PATH                         Root folder for experiment results (default: results)
--mode {single|multi|both}            Experiment type (default: multi)
                                      - single: Only single-agent baselines
                                      - multi:  Only multi-agent coordination
                                      - both:   Both single and multi-agent
--personas P1 P2 ...                  Specific personas to run (default: all 5)
--coordination C1 C2 ...              Coordination mechanisms to test (default: all 3)
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

After running experiments, results are organized in timestamped experiment folders.

See [experiment_folder_structure.md](experiment_folder_structure.md) for detailed documentation.

Explore results:

```bash
jupyter notebook 01_Result_Analysis.ipynb
```

## Publications & Citation

This system was developed for:
- **Course:** Advanced Seminar Information Management (WS 25/26)
- **Institution:** University of Cologne

## License

[LICENSE](LICENSE) — See LICENSE file for details.