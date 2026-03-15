# Agent Architecture

This document describes the investor agent workflow implemented in this project.

## Pipeline Overview

### Single-Agent Pipeline

The single-agent architecture follows a **3-node LangGraph pipeline** that mirrors a systematic investment decision process for a single investor persona:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           SIMULATION LOOP                                 │
│  For each period:                                                         │
│                                                                           │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────┐ │
│  │   (Market    │    │  screening   │───▶│   analysis   │───▶│decision│ │
│  │    Data)     │───▶│  (Step 1)    │    │  (Step 2)    │    │(Step 3)│ │
│  │              │    │              │    │              │    │        │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └────────┘ │
│  (Portfolio State)      (LLM Call)         (LLM Call)       (LLM Call) │
│                                                                 │        │
│                         Memory Context ──────────────────────> │        │
│                                                                 │        │
│                                                                 ▼        │
│                                                          ┌────────────┐ │
│                                                          │  EXECUTION │ │
│                                                          │ (Portfolio)│ │
│                                                          └────────────┘ │
└──────────────────────────────────────────────────────────────────────────┘
```

### Multi-Agent Pipeline Extension

The multi-agent architecture extends the single-agent pipeline with a two-pass coordination mechanism:

```
PASS 1 (Parallel for each persona)        PASS 2 (Parallel for each persona)
┌──────────────────────────────┐         ┌──────────────────────────────┐
│  Steps 1-3 (Single-Agent)    │         │  Step 5 (Peer-Informed)     │
│  ┌─────────┐ ┌────────┐     │         │  ┌──────────┐ ┌──────────┐  │
│  │Screening│→│Analysis│─┐   │         │  │Re-Anal.  │→│Revision  │  │
│  └─────────┘ └────────┘ │   │         │  └──────────┘ └──────────┘  │
│                         └─┐ │         │         ▲                     │
│                           ▼ │         │         │                     │
│                      ┌───────┐         │   Decision Summary             │
│                      │Decision         │   (from Pass 1)               │
│                      └───────┘         │                               │
│ (for each agent)            │          │ (for each agent)              │
└─────────────────────────────┼──────────┼─────────────────────────────┘
                              │          │
                    ┌─────────▼──────────▼──────────┐
                    │  Step 4: Summary (Build)      │
                    │  Step 7: Coordination Layer   │
                    │  (majority_vote/avg_size/     │
                    │   llm_manager)                │
                    └─────────┬──────────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │ Step 8: Execution  │
                    │   (Portfolio)      │
                    └────────────────────┘
```

---

## Single-Agent Node Descriptions

### Node 1: screening (LLM Node - Step 1: Market Screening)

**Purpose:** Screen the market for investment candidates based on persona philosophy.

**Input from State:**
- `portfolio`: Current holdings (dict mapping ticker → quantity)
- `cash`: Available cash
- `market_universe`: List of dicts with ticker and fundamentals
- `period_label`: Current period identifier
- `persona_prompt`: Persona's system prompt

**Prompt Template:**
```
{system_prompt}

Step 1: SCREENING for {period_label}.
{memory_context}

RULES:
- recheck_tickers: current holdings to re-evaluate.
- candidate_tickers: 8-12 best candidates per your philosophy.
- Output ONLY JSON:
{"recheck_tickers":["t0001"],"candidate_tickers":["t0002","t0003"],"rationale":"..."}
```

**Output Schema:**
```python
class ScreeningOutput(BaseModel):
    recheck_tickers: list[str]      # Current holdings to re-evaluate
    candidate_tickers: list[str]    # 8-12 tickers for deep analysis
    rationale: str                   # One sentence rationale
```

**Retry Logic:** MAX_RETRIES=2, delay=2s. Falls back to holding all positions + top 10 candidates.

---

### Node 2: analysis (LLM Node - Step 2: Fundamental Analysis)

**Purpose:** Deep-dive analysis of selected candidates using detailed financial metrics.

**Input from State:**
- `screening`: Output from Node 1 (recheck + candidate tickers)
- `portfolio`: Current holdings
- `cash`: Available cash
- `period_label`: Period identifier
- `persona_prompt`: Persona system prompt
- `memory_context`: Historical trade and position context

**Process:**
1. Load fundamentals for all candidates from `fundamentals_db`
2. Extract 20+ key metrics: pe_ratio, pb_ratio, roe, roe, revenue_growth, etc.
3. Truncate large numbers for readability
4. Invoke LLM with persona prompt + fundamentals JSON

**Output Schema:**
```python
class CompanyAnalysis(BaseModel):
    ticker: str                          # Ticker code
    thesis: str                          # One-sentence investment thesis (max 20 words)
    key_metrics: dict[str, Any]          # 3-5 key metrics with values
    signal: str                          # STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL
    conviction: int                      # Conviction level 1-10

class AnalysisOutput(BaseModel):
    analyses: list[CompanyAnalysis]
```

**Retry Logic:** MAX_RETRIES=2. Falls back to empty analysis list.

**Key Metrics Extracted:**
- Valuation: pe_ratio, pb_ratio, market_cap, enterprise_value
- Profitability: roe, roa, net_income_margin, revenue, net_income
- Balance Sheet: current_ratio, debt_to_equity, cash_equivalents
- Growth: revenue_growth_yoy
- Graham indicators: ncav_per_share, book_value_per_share
- Price data: open, high, low, close, volume

---

### Node 3: decision (LLM Node - Step 3: Trade Decisions)

**Purpose:** Produce final structured trading orders based on analysis.

**Input from State:**
- `analyses`: Output from Node 2 with signals and conviction
- `portfolio`: Current holdings + share counts
- `cash`: Available cash
- `period_label`: Period identifier
- `persona_prompt`: Persona system prompt
- `memory_context`: Past rationales and period returns

**Prompt Template:**
```
{system_prompt}

Step 3: TRADE DECISIONS for {period_label}.
{memory_context}

RULES:
1. For each analysed ticker decide: BUY, SELL, or HOLD.
2. HOLD means no action — quantity must be 0.
3. Valid to HOLD all if no improvement possible.
4. Each order: ticker, action, quantity (int≥0), reasoning (max 15 words).
5. BUY cost ≤ cash. SELL qty ≤ held shares.
6. Size by conviction: highest conviction = largest position.
7. Output ONLY JSON:
{"orders":[{...}],"portfolio_rationale":"One sentence."}
```

**Output Schema:**
```python
class TradeOrder(BaseModel):
    ticker: str                    # Ticker code
    action: str                    # BUY, SELL, or HOLD
    quantity: int                  # Shares (0 required for HOLD)
    reasoning: str                 # Max 15 words

class DecisionOutput(BaseModel):
    orders: list[TradeOrder]
    portfolio_rationale: str       # One sentence summary
```

**LLM Config:** Temperature=0.1 (low for consistency)
**Retry Logic:** MAX_RETRIES=2. Falls back to empty order list.

---

## AgentMemory System

Each agent maintains an `AgentMemory` instance across all periods to provide historical context:

```python
@dataclass
class TradeRecord:
    period: str; ticker: str; action: str; quantity: int; price: float; reasoning: str

@dataclass
class PositionMemory:
    ticker: str; entry_period: str; entry_price: float; current_shares: int
    cost_basis: float; unrealized_pnl: float; periods_held: int

@dataclass
class AgentMemory:
    trade_history: list[TradeRecord]                  # All trades executed
    position_tracker: dict[str, PositionMemory]      # Current positions with history
    period_returns: list[dict]                        # [{"period", "value", "return_pct"}, ...]
    past_rationales: list[dict]                       # [{"period", "rationale"}, ...]
    
    def to_prompt_context() -> str:
        # Converts memory into readable prompt context for next period
```

This context is injected into each LLM call via `memory_context` state field.

---

## State Schema

The `AgentState` TypedDict flows through all nodes:

```python
class AgentState(TypedDict, total=False):
    # Portfolio state
    portfolio: dict                # Holdings: {ticker → quantity}
    cash: float                    # Available cash balance
    
    # Market data
    market_universe: list          # List of {ticker, fundamentals, ...} dicts
    
    # Context
    period_label: str              # Current period identifier
    persona_prompt: str            # Persona's system prompt
    memory_context: str            # Historical trades + positions + returns
    
    # Pipeline outputs
    screening: dict                # ScreeningOutput (recheck + candidates)
    analyses: dict                 # AnalysisOutput (list of CompanyAnalysis)
    decision: dict                 # DecisionOutput (orders + rationale)
    
    # Error handling
    error: str                     # Optional error message
```

---

## Multi-Agent Extension

### Pass 1: Independent Agent Decisions (Parallel)

All personas run Steps 1-3 independently in parallel:
```python
# For each persona:
app = build_agent_graph(fundamentals_db, price_data)
final_state = app.invoke(initial_state)  # Steps 1-3
# Result: first_pass_states[persona] = {screening, analyses, decision}
```

### Step 4: Decision Summary

After Pass 1, build a summary of all agents' decisions:

```python
def build_decision_summary(agent_results: dict[str, dict]) -> str:
    {
        "period": "2020-01",
        "agent_decisions": [
            {
                "persona": "warren_buffett",
                "orders": [{ticker, action, quantity, reasoning}, ...],
                "top_signals": ["STRONG_BUY", "BUY", ...] (top 8)
            },
            ...
        ]
    }
```

This summary becomes input to Pass 2.

---

### Pass 2: Peer-Informed Re-Analysis (Parallel)

Each persona re-analyzes with awareness of peer decisions:

#### Node 5: reanalysis_with_peers (LLM Node)

**Input:**
- Peer decision summary from Step 4
- Same candidates as Pass 1
- Persona's original portfolio state
- Persona prompt and memory

**Prompt:**
```
You have peer agent decisions below. 
Review them but maintain your own philosophy.
...
Output the same analysis schema.
```

**Output:** Revised `AnalysisOutput` with potentially adjusted signals/conviction.

#### Node 6: revised_decision (LLM Node)

Same as Step 3 but using revised analysis from node 5.
**Output:** Revised `DecisionOutput`

### Build Pass 2 Graph:
```python
def build_pass2_graph(fundamentals_db, price_data, decision_summary):
    # reanalysis → (conditional) → redecision → END
```

---

## Step 7: Coordination Mechanisms

After Pass 2 completes, aggregate all agents' revised decisions using one of:

### Majority Vote Coordination
```python
class MajorityVoteCoordination:
    # For each ticker:
    #   - Count BUY votes vs SELL votes
    #   - If > n_agents/2 votes for BUY: create BUY order (avg quantity)
    #   - If > n_agents/2 votes for SELL: create SELL order (avg quantity)
```

### Average Size Coordination
```python
class AverageSizeCoordination:
    # For each ticker:
    #   - Net = #BUY agents - #SELL agents
    #   - If net > 0: BUY with avg quantity from BUY agents
    #   - If net < 0: SELL with avg quantity from SELL agents
```

### LLM Manager Coordination
```python
class LLMManagerCoordination:
    # Invoke LLM with all proposals + portfolio state
    # LLM synthesizes into final orders respecting:
    #   - Cash constraints
    #   - Diversification
    #   - Consensus-seeking
    # Output: Final DecisionOutput
```

All mechanisms respect:
- Cash constraints: `total_cost(buys) <= portfolio.cash`
- Position constraints: `SELL qty <= held_shares`
- Minimum 1 order output (empty fallback to Majority Vote)

---

## Step 8: Execution

Execute final orders through Portfolio class:

```python
# 1. Apply trades
trade_log = portfolio.apply_trades(final_orders, price_data)

# 2. Update each agent's memory with executed trades
for persona in persona_names:
    memory[persona].record_trade(period, order, exec_price)

# 3. Snapshot portfolio state
portfolio.snapshot(period, price_data)

# 4. Update agent return statistics
for memory in agent_memories.values():
    memory.update_period_end(period, portfolio_value, price_data, initial_capital)
```

---

## Execution Configuration

### Single-Agent Backtest
```python
def run_backtest(
    persona_name: str,      # Single persona
    periods: list[str],
    market_universe_by_period: dict,
    fundamentals_by_period: dict,
    prices_by_period: dict,
    n_runs: int = 3,        # Number of replications
    ...
) -> list[dict]            # Results per run
```

### Multi-Agent Backtest
```python
def run_multiagent_backtest(
    persona_names: list[str],  # Multiple personas
    periods: list[str],
    market_universe_by_period: dict,
    fundamentals_by_period: dict,
    prices_by_period: dict,
    coordination: str = "majority_vote",  # "majority_vote" | "average_size" | "llm_manager"
    n_runs: int = 3,
    workers: int = 1,       # Parallel agents per pass
    ...
) -> list[dict]            # Results with coordination breakdown
```

---

## LLM Configuration

The agent uses **Google Gemini 2.5 Flash Lite** with:
- **Model:** `gemini-2.5-flash-lite`
- **Screening & Analysis Temperature:** 0.5 (balanced)
- **Decision Temperature:** 0.1 (conservative/consistent)
- **Retry:** MAX_RETRIES=2, RETRY_DELAY_SEC=2.0
- **Structured Output:** Pydantic schema enforcement with JSON parsing

```python
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=temperature,
    google_api_key=os.environ.get("GOOGLE_API_KEY"),
    convert_system_message_to_human=False
)
```

---

## Data Flow Architecture

```
Input Data (Per Period)
─────────────────────────────────
    market_universe: [{ticker, fundamentals}, ...]
    price_data: {ticker → price}
    fundamentals_db: {ticker → {metric → value}}
                  │
                  ▼
    ┌─────────────────────────────┐
    │   PASS 1: Single-Agent Run  │
    │   (Steps 1-3, parallizable) │
    └──────────────┬──────────────┘
    Per Persona:   │
    ├─ Node 1: screening → ScreeningOutput
    ├─ Node 2: analysis → AnalysisOutput
    └─ Node 3: decision → DecisionOutput
                  │
                  ├─ first_pass_states[persona]
                  │
                  ▼
    ┌─────────────────────────────┐
    │  Step 4: Build Summary      │
    │  (All agents' decisions)    │
    └──────────────┬──────────────┘
                  │
                  ▼
    ┌─────────────────────────────┐
    │   PASS 2: Peer-Informed     │
    │   (Steps 5-6, parallizable) │
    └──────────────┬──────────────┘
    Per Persona:   │
    ├─ Node 5: reanalysis_with_peers → revised AnalysisOutput
    └─ Node 6: revised_decision → revised DecisionOutput
                  │
                  ├─ second_pass_states[persona]
                  │
                  ▼
    ┌─────────────────────────────────┐
    │  Step 7: Coordination Mechanism │
    │  (majority_vote/avg_size/       │
    │   llm_manager)                  │
    └──────────────┬──────────────────┘
                  │
                  ▼
           Final Trade Orders
                  │
                  ▼
    ┌─────────────────────────────┐
    │  Step 8: Portfolio Execution│
    │  Execute trades in order    │
    └──────────────┬──────────────┘
                  │
                  ▼
    Snapshot & Update Memories
```

---

## Persona Configuration

Each persona is defined in [personas.py](personas.py):

```python
PERSONAS = {
    "warren_buffett": {system_prompt},
    "cathie_wood": {system_prompt},
    "joel_greenblatt": {system_prompt},
    "ben_graham": {system_prompt},
    "buffett": {system_prompt},
}
```

- **system_prompt:** Character-specific instructions injected into every LLM call
- All personas use the same node structure (Steps 1-3) but with different reasoning

---

## Key Implementation Details

### Error Handling & Retry
- **MAX_RETRIES = 2** with **RETRY_DELAY_SEC = 2.0**
- Each node has a fallback:
  - Screening fallback: Hold all + random 10 candidates
  - Analysis fallback: Empty analyses list
  - Decision fallback: Empty orders (no action)
- Logged failures include attempt count and error message

### JSON Parsing Robustness
- Primary: Direct `json.loads()`
- Fallback 1: Strip markdown code blocks
- Fallback 2: Regex extraction of JSON objects/arrays
- Fallback 3: Repair common errors (trailing commas, quote mismatches)

### Performance Optimization
- **Parallel Execution:** Multi-agent Pass 1 & Pass 2 can run agents in parallel via `ThreadPoolExecutor(max_workers=workers)`
- **Shared Data:** `fundamentals_db` and `price_data` passed to graph builders to avoid state bloat
- **Memory Truncation:** Large financial numbers (>1,000,000) are abbreviated for readability

### Logging
All LLM calls logged via `ExperimentLogger`:
- System prompt, human prompt, raw output, parsed output, success/error
- Temperature recorded per call
- Enables auditability and debugging
