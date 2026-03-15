import wrds
import os
import pandas as pd
from dotenv import load_dotenv

class WRDSDATA():
    # --------------------
    # need to have pgpass file set up for passwordless access
    # --------------------
    
    def __init__(self):
        self.conn = None

        load_dotenv() 
        self.WRDS_USERNAME = os.getenv("WRDS_USERNAME")


    def __enter__(self):
        self.conn = wrds.Connection(wrds_username=self.WRDS_USERNAME)
        return self 

    def __exit__(self, exc_type, exc_value, traceback):
        if self.conn:
            self.conn.close()
        # return False to propagate exceptions
        return False


    def get_nasdaq100_comp_asof(self, date) -> 'pd.DataFrame':
        """
        Pull Nasdaq-100 companies identified by gvkey and iid at specific date.

        date: str in 'YYYY-MM-DD' format
        """
        sql = """
            SELECT
                c.gvkey,
                c.iid
            FROM comp.idxcst_his AS c
            WHERE 
                c.gvkeyx = '000208'
                AND %(date)s BETWEEN c."from" AND COALESCE(c.thru, CURRENT_DATE)
            ORDER BY c.gvkey, c."from"
        """
        df = self.conn.raw_sql(sql, params={"date": date})
        return df


    def get_nasdaq100_crsp_asof(self, date) -> 'pd.DataFrame':
        """
        Get CRSP permnos and corresponding gvkeys, iids, and primary tickers as of a specific date.
        Permnos are unique identifiers for securities in CRSP database.

        date: str in 'YYYY-MM-DD' format
        
        Returns
        -------
        pd.DataFrame
            Columns: gvkey, iid, permno, ticker
        """
        # Step 1: get Nasdaq-100 gvkeys and iids
        gvkeys_df = self.get_nasdaq100_comp_asof(date)
        if gvkeys_df.empty:
            return gvkeys_df

        gvkeys = gvkeys_df['gvkey'].unique().tolist()
        
        # Step 2: query CRSP link table for permnos
        sql = """
            SELECT 
                gvkey,
                liid,
                lpermno,
                linkprim
            FROM crsp.ccmxpf_linktable
            WHERE 
                gvkey = ANY(%(gvkeys)s)
                AND linkdt <= DATE %(date)s
                AND COALESCE(linkenddt, CURRENT_DATE) >= DATE %(date)s
                AND linkprim IN ('P','C')
        """
        permno_df = self.conn.raw_sql(sql, params={"gvkeys": gvkeys, "date": date})

        # Step 3: merge with gvkey/iid info
        df = gvkeys_df.merge(
            permno_df,
            left_on=['gvkey', 'iid'],
            right_on=['gvkey', 'liid'],
            how='inner'
        )

        df['permno'] = df['lpermno'].astype('Int64')
        df = df.drop(columns=['liid','linkprim', 'lpermno'])
        
        # Step 4: Add primary ticker symbols using crsp.stocknames
        if not df.empty:
            permnos = df['permno'].dropna().unique().tolist()
            ticker_sql = """
                WITH primary_securities AS (
                    SELECT 
                        permno, 
                        ticker,
                        exchcd,
                        shrcd,
                        shrcls,
                        namedt,
                        nameenddt,
                        -- Prioritize: Major exchanges, common shares, primary share class, most recent
                        ROW_NUMBER() OVER (
                            PARTITION BY permno 
                            ORDER BY 
                                CASE WHEN exchcd IN (1,2,3) THEN 1 ELSE 2 END,  -- NYSE, AMEX, NASDAQ
                                CASE WHEN shrcd IN (10,11) THEN 1 ELSE 2 END,    -- Common shares
                                CASE WHEN shrcls IS NULL OR shrcls = '' OR shrcls = 'A' THEN 1 ELSE 2 END,  -- Primary class
                                nameenddt DESC NULLS FIRST, 
                                namedt DESC
                        ) as rn
                    FROM crsp.stocknames
                    WHERE permno = ANY(%(permnos)s)
                      AND namedt <= %(date)s
                      AND COALESCE(nameenddt, CURRENT_DATE) >= %(date)s
                )
                SELECT permno, ticker
                FROM primary_securities
                WHERE rn = 1
            """
            tickers_df = self.conn.raw_sql(ticker_sql, params={"permnos": [int(p) for p in permnos], "date": date})
            
            if not tickers_df.empty:
                tickers_df['permno'] = tickers_df['permno'].astype('Int64')
                df = df.merge(tickers_df, on='permno', how='left')
            else:
                df['ticker'] = None
        
        return df

    def _get_fundamentals_for_gvkeys(self, gvkeys: list, date: str) -> 'pd.DataFrame':
        """
        Internal: fetch quarterly fundamentals for given gvkeys.
        
        Returns DataFrame with gvkey as key column for joining.
        """
        if not gvkeys:
            return pd.DataFrame()

        sql = """
            WITH candidates AS (
                SELECT
                    gvkey, 
                    rdq as report_date,
                    fyearq as fiscal_year,
                    cshoq as shares_outstanding,
                    atq as assets,
                    lctq as liabilities,
                    revtq as revenue,
                    niq as net_income,
                    cheq as cash_equivalents,
                    dvy as dividends_yearly,
                    ppentq as net_ppe,
                    ROW_NUMBER() OVER (
                        PARTITION BY gvkey
                        ORDER BY datadate DESC
                    ) AS rn
                FROM comp.fundq
                WHERE gvkey = ANY(%(gvkeys)s)
                    AND indfmt = 'INDL'
                    AND datafmt = 'STD'
                    AND consol = 'C'
                    AND popsrc = 'D'
                    AND datadate <= DATE %(date)s
                    AND (rdq IS NULL OR rdq <= DATE %(date)s)
                    AND datadate >= DATE %(date)s - INTERVAL '5 years'
            )
            SELECT *
            FROM candidates
            WHERE rn = 1
            ORDER BY gvkey;
        """
        df = self.conn.raw_sql(sql, params={"gvkeys": gvkeys, "date": date})
        if not df.empty:
            df.drop(columns=['rn'], inplace=True)
        return df

    def get_fundamentals_qrtly_asof(self, date) -> 'pd.DataFrame':
        """
        Pull latest Compustat quarterly fundamentals for all gvkeys as of a specific date.
        
        date: str in 'YYYY-MM-DD' format
        """
        gvkeys_df = self.get_nasdaq100_crsp_asof(date)
        if gvkeys_df.empty:
            return gvkeys_df

        gvkeys = gvkeys_df['gvkey'].unique().tolist()
        fundamentals = self._get_fundamentals_for_gvkeys(gvkeys, date)
        
        if fundamentals.empty:
            return gvkeys_df
        
        return gvkeys_df.merge(fundamentals, on='gvkey', how='left')

    def _get_period_ohlcv_for_permnos(self, permnos: list, date: str, window_days: int = 10) -> 'pd.DataFrame':
        """
        Internal: fetch and aggregate OHLCV for given permnos over a time period.
        
        Aggregates OHLCV data across the specified window:
        - Open: first open price from the period
        - High: maximum high price across the period
        - Low: minimum low price across the period
        - Close: last close price from the period
        - Volume: total volume across the period
        
        Parameters
        ----------
        permnos : list
            List of CRSP permno identifiers
        date : str
            End date in 'YYYY-MM-DD' format
        window_days : int, default 10
            Number of days to look back for aggregation
        
        Returns
        -------
        pd.DataFrame
            DataFrame with permno as key column and aggregated OHLCV data
        """
        if not permnos:
            return pd.DataFrame()

        sql_end = date
        sql_start = (pd.Timestamp(date) - pd.Timedelta(days=window_days)).strftime("%Y-%m-%d")
        permnos_list = [int(p) for p in permnos]

        sql = """
            SELECT
                permno,
                date,
                prc,
                openprc,
                bidlo,
                askhi,
                vol,
                cfacpr
            FROM crsp.dsf
            WHERE permno = ANY(%(permnos)s)
            AND date BETWEEN %(start_date)s AND %(end_date)s
            ORDER BY permno, date DESC
        """

        df = self.conn.raw_sql(sql, params={
            "permnos": permnos_list,
            "start_date": sql_start,
            "end_date": sql_end
        })
        if df.empty:
            return df

        df['permno'] = df['permno'].astype('Int64')
        
        # Construct raw OHLCV first, then aggregate across time period
        df['close'] = df['prc'].where(df['prc'].notna(), (df['bidlo'] + df['askhi']) / 2)
        df['open'] = df['openprc'].fillna(df['close'])
        df['high'] = df['askhi'].fillna(df['close'])
        df['low'] = df['bidlo'].fillna(df['close'])
        df['volume'] = df['vol']

        # Apply cumulative factor adjustment
        df['adj_close'] = df['close'] / df['cfacpr']
        df['adj_open'] = df['open'] / df['cfacpr']
        df['adj_high'] = df['high'] / df['cfacpr']
        df['adj_low'] = df['low'] / df['cfacpr']
        df['adj_volume'] = df['volume'] * df['cfacpr']

        # Sort by permno and date to ensure proper aggregation
        df = df.sort_values(['permno', 'date'], ascending=[True, True])
        
        # Aggregate OHLCV over the time period for each permno
        aggregated = df.groupby('permno', as_index=False).agg({
            'date': 'last',  # Use the latest date
            'adj_open': 'first',  # First open (earliest date)
            'adj_high': 'max',    # Highest high across period
            'adj_low': 'min',     # Lowest low across period
            'adj_close': 'last',  # Last close (latest date)
            'adj_volume': 'sum'   # Total volume across period
        })
        
        # Rename columns to match expected output
        df = aggregated.rename(columns={
            'adj_open': 'open',
            'adj_high': 'high', 
            'adj_low': 'low',
            'adj_close': 'close',
            'adj_volume': 'volume'
        })
        
        return df

    def get_period_ohlcv_asof(self, date, window_days=10):
        """
        Construct aggregated OHLCV for Nasdaq-100 permnos over a time period ending at date.

        Aggregates OHLCV data across the specified window:
        - Open: first open price from the period
        - High: maximum high price across the period
        - Low: minimum low price across the period
        - Close: last close price from the period
        - Volume: total volume across the period

        Uses CRSP DSF columns:
            prc, openprc, bidlo, askhi, vol, cfacpr, dlstcd, etc.

        Adjusts for cumulative factors and uses data from window_days <= date.
        Does not include delisting returns.

        Parameters
        ----------
        date : str
            End date in 'YYYY-MM-DD' format
        window_days : int, default 10
            Number of days to look back for aggregation

        Returns
        -------
        pd.DataFrame
            One row per permno with aggregated OHLCV data
        """
        gv_df = self.get_nasdaq100_crsp_asof(date)
        if gv_df.empty:
            return gv_df

        gv_df['permno'] = gv_df['permno'].astype('Int64')
        permnos = gv_df['permno'].dropna().unique().tolist()
        
        ohlcv = self._get_period_ohlcv_for_permnos(permnos, date, window_days)
        if ohlcv.empty:
            return ohlcv

        return gv_df.merge(ohlcv, on='permno', how='left')

    def get_market_data_asof(self, date: str, window_days: int = 10) -> 'pd.DataFrame':
        """
        Get combined fundamentals and OHLCV for Nasdaq-100 as of a date.
        
        Fetches the universe once, then left joins fundamentals and prices.
        Returns one row per primary ticker.
        
        Parameters
        ----------
        date : str
            Date in 'YYYY-MM-DD' format.
        window_days : int
            Lookback window for OHLCV prices.
            
        Returns
        -------
        pd.DataFrame
            Combined data with columns from universe, fundamentals, and OHLCV.
        """
        # 1. Get the base universe (one row per primary ticker)
        universe = self.get_nasdaq100_crsp_asof(date)
        if universe.empty:
            return universe
        
        universe['permno'] = universe['permno'].astype('Int64')
        gvkeys = universe['gvkey'].unique().tolist()
        permnos = universe['permno'].dropna().unique().tolist()
        
        # 2. Fetch fundamentals and OHLCV separately
        fundamentals = self._get_fundamentals_for_gvkeys(gvkeys, date)
        ohlcv = self._get_period_ohlcv_for_permnos(permnos, date, window_days)
        
        # 3. Left join to base universe
        result = universe
        if not fundamentals.empty:
            result = result.merge(fundamentals, on='gvkey', how='left')
        if not ohlcv.empty:
            result = result.merge(ohlcv, on='permno', how='left')
        
        return result
    
    def period_buy_prices_asof(self, date, window_days=10):
        """
        Get aggregated closing prices for Nasdaq-100 securities over a time period.
        
        Uses the close price from period-aggregated OHLCV data, which represents
        the last closing price from the aggregation window.
        
        Parameters
        ----------
        date : str
            End date in 'YYYY-MM-DD' format
        window_days : int, default 10
            Number of days to look back for aggregation
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns 'permno', 'date', 'price'
        """
        df = self.get_period_ohlcv_asof(date, window_days)

        return df[[
            'permno',
            'date',
            'close'
        ]].rename(columns={'close': 'price'})

    def tickers_to_permnos(self, tickers: list[str], date: str) -> pd.DataFrame:
        """
        Convert a list of CRSP tickers to their permnos as of a specific date.

        Parameters
        ----------
        tickers : list of str
            Ticker symbols to look up
        date : str
            Date in 'YYYY-MM-DD' format

        Returns
        -------
        pd.DataFrame
            DataFrame with columns 'permno' and 'ticker'
        """
        sql = """
            SELECT permno, ticker
            FROM crsp.stocknames
            WHERE ticker = ANY(%(tickers)s)
            AND namedt <= %(date)s
            AND COALESCE(nameenddt, CURRENT_DATE) >= %(date)s
        """
        return self.conn.raw_sql(sql, params={"tickers": tickers, "date": date})

    def get_prices_for_permnos(self, permnos: list[int], date: str, window_days: int = 10) -> pd.DataFrame:
        """
        Get prices for specific permnos as of a date.

        Parameters
        ----------
        permnos : list of int
            CRSP permno identifiers
        date : str
            Date in 'YYYY-MM-DD' format
        window_days : int
            Lookback window for finding last available price

        Returns
        -------
        pd.DataFrame
            DataFrame with columns 'permno' and 'price'
        """
        if not permnos:
            return pd.DataFrame(columns=["permno", "price"])

        sql_start = (pd.Timestamp(date) - pd.Timedelta(days=window_days)).strftime("%Y-%m-%d")
        permnos_list = [int(p) for p in permnos]

        sql = """
            WITH ranked AS (
                SELECT
                    permno,
                    prc,
                    cfacpr,
                    ROW_NUMBER() OVER (PARTITION BY permno ORDER BY date DESC) as rn
                FROM crsp.dsf
                WHERE permno = ANY(%(permnos)s)
                AND date BETWEEN %(start_date)s AND %(end_date)s
            )
            SELECT permno, ABS(prc) / cfacpr as price
            FROM ranked
            WHERE rn = 1
        """
        return self.conn.raw_sql(sql, params={
            "permnos": permnos_list,
            "start_date": sql_start,
            "end_date": date
        })

    # ──────────────────────────────────────────────────────────────────────
    # Index Daily Prices (Compustat)
    # ──────────────────────────────────────────────────────────────────────
    # MSCI World:   comp.g_idx_daily, gvkeyx='150066', prccd column
    # NASDAQ-100:   comp.idx_daily, gvkeyx='000208', prccd column

    MSCI_WORLD_GVKEYX = '150066'
    NASDAQ100_GVKEYX = '000208'

    def get_msci_world_index_prices(
        self, start_date: str, end_date: str,
    ) -> pd.DataFrame:
        """
        Get MSCI World index daily closing prices from Compustat Global.

        Parameters
        ----------
        start_date, end_date : str
            Date range in 'YYYY-MM-DD' format.

        Returns
        -------
        pd.DataFrame
            Columns: datadate (date), price (float).
            Sorted by datadate ascending.
        """
        sql = """
            SELECT datadate, prccd AS price
            FROM comp.g_idx_daily
            WHERE gvkeyx = %(gvkeyx)s
              AND datadate BETWEEN %(start)s AND %(end)s
              AND prccd IS NOT NULL
            ORDER BY datadate
        """
        return self.conn.raw_sql(sql, params={
            "gvkeyx": self.MSCI_WORLD_GVKEYX,
            "start": start_date,
            "end": end_date,
        })

    def get_nasdaq100_index_prices(
        self, start_date: str, end_date: str,
    ) -> pd.DataFrame:
        """
        Get NASDAQ-100 index daily closing prices from Compustat.

        Parameters
        ----------
        start_date, end_date : str
            Date range in 'YYYY-MM-DD' format.

        Returns
        -------
        pd.DataFrame
            Columns: datadate (date), price (float).
            Sorted by datadate ascending.
        """
        sql = """
            SELECT datadate, prccd AS price
            FROM comp.idx_daily
            WHERE gvkeyx = %(gvkeyx)s
              AND datadate BETWEEN %(start)s AND %(end)s
              AND prccd IS NOT NULL
            ORDER BY datadate
        """
        return self.conn.raw_sql(sql, params={
            "gvkeyx": self.NASDAQ100_GVKEYX,
            "start": start_date,
            "end": end_date,
        })