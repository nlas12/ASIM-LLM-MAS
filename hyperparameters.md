## Hyperparameters

### MAX_NEW_CANDIDATES = 10 
- Ensures the agent doesn't over-analyze by capping new stocks at 10.
- If more than 10 stocks have been selected by the LLM, we take the first 10.
- Current holdings are always fully re-evaluated.

### MAX_RETRIES = 2, RETRY_DELAY=2.0
- number of retries if any step fails
- if retries exceed MAX_RETRIES use fallback methods, that return either all tickers currently in the portfolio for screening or empty sets of tickers for analysis/decision steps.

### LLM_TEMPERATURE=0.5, DECISION_TEMPERATURE=0.1
- different temperatures between screening and decision so that decisions dont deviate significantly from screening results

### Number of Personas = 5
- arbitrary number

### Model
- used Gemini 2.5 Flash (Lite)

### INITIAL_CAPITAL = 1_000_000.0
- arbitrary number

## Hyperparameter-adjacent

### Persona Prompts
- adapted from Guru-Agents
- extended by 2 personas: cathie_wood, ray_dalio

### Coordination Mechanisms
- **Majority Vote**: For each ticker, counts agent BUY/SELL votes. If >50% of agents vote the same direction, executes that trade with average quantity.
- **Average Size**: Calculates net direction per ticker (buy count - sell count). If net is positive, buys with average quantity; if negative, sells with average quantity.
- **LLM Manager**: Passes all agent proposals to an LLM "portfolio manager" which synthesizes them into final orders, resolving conflicts via diversification and staying within cash constraints. Falls back to majority vote on LLM failure.

### Trading Intervals = Quarterly
- Chosen over monthly cadence to reduce the number of trading periods in one simulation run.

### Universe (NASDAQ100)
- nasdaq mainly includes tech stocks
- opted for nasdaq instead of SP500 for availability, and NASDAQ100 over NASDAQ500 for speed

### Metrics

#### window_days=10 (in wrds_data.py)
- OHLCV is averaged, or respectively max() min() is taken over that window
- newest available market data is searched in [date-window_days;date]

#### Performance Metrics (Risk & Return)
- **total_return_pct**: Cumulative return as percentage
- **annualized_return_pct** (CAGR): Compound annual growth rate
- **annualized_volatility_pct**: Annualized standard deviation of returns
- **sharpe_ratio**: (Mean excess return) / volatility, risk-adjusted return
- **sortino_ratio**: (Mean excess return) / downside deviation, downside risk-adjusted return
- **max_drawdown_pct**: Peak-to-trough percentage decline
- **final_portfolio_value**: Ending portfolio value
- **n_periods**: Number of quarterly periods in backtest

#### Valuation Metrics
- **pe_ratio**: Price-to-earnings ratio
- **pb_ratio**: Price-to-book ratio
- **market_cap**: Market capitalization
- **enterprise_value**: Market cap + debt - cash

#### Profitability Metrics
- **roe**: Return on equity
- **roa**: Return on assets
- **net_income_margin**: Net income / revenue
- **revenue**: Total revenue
- **net_income**: Net income / earnings
- **operating_income**: EBIT / operating earnings

#### Balance Sheet Metrics
- **assets**: Total assets
- **liabilities**: Total liabilities
- **cash_equivalents**: Cash and equivalents
- **current_ratio**: Current assets / current liabilities
- **debt_to_equity**: Total debt / equity
- **current_assets**: Current assets
- **current_liabilities**: Current liabilities
- **long_term_debt**: Long-term debt

#### Growth Metrics
- **revenue_growth_yoy**: Year-over-year revenue growth

#### Graham Value Investing Metrics
- **ncav**: Net current asset value (current assets - total liabilities)
- **ncav_per_share**: NCAV divided by shares outstanding
- **book_value_per_share**: Equity / shares outstanding

#### Greenblatt Magic Formula Metrics
- **earnings_yield**: EBIT / enterprise value
- **return_on_capital**: EBIT / net working capital

#### Price Data Metrics
- **open**: Opening price
- **high**: High price
- **low**: Low price
- **close**: Closing price
- **volume**: Trading volume