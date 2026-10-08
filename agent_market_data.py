# =============================================================================
# SELF-DRIVING PORTFOLIO: FREE REAL-DATA DOWNLOADER
# =============================================================================
# Builds the 10 input dataframes from free public sources, as a drop-in
# replacement for the synthetic data generator in the notebook:
#
#   FRED (St. Louis Fed)    macro series, CPI, Treasury yields, credit spreads,
#                           Chicago Fed financial conditions index
#   Yahoo Finance           daily prices of the 18 assets (via yfinance)
#   Kenneth French library  daily Fama-French 3 factors
#   Robert Shiller data     S&P 500 CAPE, earnings yield, dividend yield
#   data/manual_inputs.xlsx survey expected returns and equity valuations,
#                           entered by hand (no free API exists)
#
# Terms of use: FRED and the French and Shiller data are free for research
# with attribution. Some FRED series are third-party data under copyright
# (the ICE BofA spreads, of which FRED only publishes recent history). Yahoo
# Finance data is for personal, non-commercial use. Everything downloaded is
# kept in data/, which git ignores, so none of it is redistributed.
# =============================================================================

from __future__ import annotations

import io
import logging
import re
import time
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger: logging.Logger = logging.getLogger(__name__)

FRED_CSV_URL: str = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
FRENCH_DAILY_URL: str = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
)
# Linked from shillerdata.com; update here if the site moves the file
SHILLER_URL: str = "https://img1.wsimg.com/blobby/go/e5e77e0b-59d1-44d9-ab25-4763ac982e53/downloads/ie_data.xls"

# Yahoo Finance symbol for each pipeline ticker. Index tickers without a free
# total return series are replaced by the matching ETF.
YAHOO_SYMBOLS: Dict[str, str] = {
    "SPTR": "^SP500TR",  # S&P 500 total return index
    "RTY": "IWM",        # Russell 2000 ETF
    "MXEA": "EFA",       # MSCI EAFE ETF
    "MXEF": "EEM",       # MSCI Emerging Markets ETF
}

# 60/40 benchmark legs: S&P 500 total return and a US aggregate bond index fund
BENCHMARK_EQUITY_SYMBOL: str = "^SP500TR"
BENCHMARK_BOND_SYMBOL: str = "VBMFX"

# FRED series ids
FRED_SERIES: Dict[str, str] = {
    "real_gdp_growth": "A191RL1Q225SBEA",  # real GDP, % change SAAR, quarterly
    "payrolls": "PAYEMS",                  # nonfarm payrolls, thousands
    "cpi_sa": "CPIAUCSL",                  # CPI-U, seasonally adjusted
    "cpi_core_sa": "CPILFESL",             # core CPI, seasonally adjusted
    "cpi_nsa": "CPIAUCNS",                 # CPI-U, not seasonally adjusted
    "brent": "DCOILBRENTEU",               # Brent, USD per barrel, daily
    "fed_funds": "FEDFUNDS",               # effective fed funds rate, monthly, %
    "fci": "NFCI",                         # Chicago Fed national financial conditions, weekly
    "ust_3m": "DGS3MO",
    "ust_2y": "DGS2",
    "ust_10y": "DGS10",
    "ust_30y": "DGS30",
    "ig_oas": "BAMLC0A0CM",                # ICE BofA US corporate OAS, %
    "hy_oas": "BAMLH0A0HYM2",              # ICE BofA US high yield OAS, %
    "em_oas": "BAMLEMCBPIOAS",             # ICE BofA EM corporate plus OAS, %
}

MANUAL_INPUTS_PATH: Path = Path("data/manual_inputs.xlsx")
EQUITY_CATEGORY: str = "Equity"


# =============================================================================
# Downloading (with a same-day cache in data/cache)
# =============================================================================

class FreeDataSources:
    """Fetches raw series from each source. Results are cached for a day under cache_dir."""

    def __init__(self, cache_dir: str | Path = "data/cache", max_age_hours: float = 20.0) -> None:
        self.cache_dir = Path(cache_dir)
        self.max_age_seconds = max_age_hours * 3600
        self.session = requests.Session()
        # FRED can stall requests that don't look like a browser, so send browser-like headers
        self.session.headers.update({
            "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
                           "(KHTML, like Gecko) Version/18.0 Safari/605.1.15"),
            "Accept": "text/csv,application/octet-stream,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        })

    def _cached(self, name: str, fetch: Callable[[], pd.DataFrame]) -> pd.DataFrame:
        path = self.cache_dir / f"{name}.csv"
        if path.exists() and time.time() - path.stat().st_mtime < self.max_age_seconds:
            return pd.read_csv(path, index_col=0, parse_dates=True)
        df = fetch()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(path)
        return df

    def _get(self, url: str, attempts: int = 4) -> bytes:
        # FRED in particular is sometimes slow or briefly unavailable: retry with growing pauses
        for attempt in range(1, attempts + 1):
            try:
                response = self.session.get(url, timeout=(15, 120))
                if response.status_code in (429, 500, 502, 503, 504) and attempt < attempts:
                    raise requests.HTTPError(f"HTTP {response.status_code}", response=response)
                response.raise_for_status()
                return response.content
            except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
                retryable = not isinstance(exc, requests.HTTPError) or (
                    exc.response is not None and exc.response.status_code in (429, 500, 502, 503, 504))
                if not retryable or attempt == attempts:
                    raise
                pause = 10 * attempt
                logger.warning("Download failed (%s); retrying in %d s (attempt %d of %d): %s",
                               type(exc).__name__, pause, attempt + 1, attempts, url)
                time.sleep(pause)
        raise RuntimeError("unreachable")

    def fred(self, series_id: str) -> pd.Series:
        def fetch() -> pd.DataFrame:
            return parse_fred_csv(self._get(FRED_CSV_URL.format(series_id=series_id)), series_id).to_frame()
        return self._cached(f"fred_{series_id}", fetch).iloc[:, 0]

    def yahoo(self, symbol: str, start: str) -> pd.DataFrame:
        def fetch() -> pd.DataFrame:
            import yfinance as yf
            df = yf.download(symbol, start=start, auto_adjust=False, actions=False,
                             progress=False, multi_level_index=False)
            if df is None or df.empty:
                raise RuntimeError(f"Yahoo Finance returned no data for {symbol}.")
            if "Adj Close" not in df.columns:
                df["Adj Close"] = df["Close"]
            df.index = pd.to_datetime(df.index).tz_localize(None)
            return df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
        safe = re.sub(r"[^A-Za-z0-9]", "_", symbol)
        return self._cached(f"yahoo_{safe}", fetch)

    def french_daily(self) -> pd.DataFrame:
        return self._cached("french_ff3_daily", lambda: parse_french_daily_zip(self._get(FRENCH_DAILY_URL)))

    def shiller(self) -> pd.DataFrame:
        return self._cached("shiller_ie_data", lambda: parse_shiller_xls(self._get(SHILLER_URL)))


# =============================================================================
# Parsers (one per source format)
# =============================================================================

def parse_fred_csv(content: bytes, series_id: str) -> pd.Series:
    """FRED graph CSV: a date column ('observation_date' or 'DATE') and one value column; '.' = missing."""
    df = pd.read_csv(io.BytesIO(content))
    date_col = df.columns[0]
    series = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    series.index = pd.to_datetime(df[date_col])
    series.index.name = "date"
    series.name = series_id
    return series.dropna()


def parse_french_daily_zip(content: bytes) -> pd.DataFrame:
    """Kenneth French daily factors: a zipped CSV with text above and below the YYYYMMDD rows, in percent."""
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        text = zf.read(zf.namelist()[0]).decode("latin-1")
    rows: List[List[str]] = []
    header: Optional[List[str]] = None
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if re.fullmatch(r"\d{8}", parts[0]):
            rows.append(parts)
        elif header is None and "Mkt-RF" in parts:
            header = parts
        elif rows:
            break  # the daily block ends at the first non-data line
    if header is None or not rows:
        raise ValueError("Could not find the daily factor table in the Kenneth French file.")
    df = pd.DataFrame(rows, columns=["date"] + header[1:])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.set_index("date").apply(pd.to_numeric, errors="coerce")
    return df.rename(columns={"Mkt-RF": "mkt_rf", "SMB": "smb", "HML": "hml", "RF": "rf"})[["rf", "mkt_rf", "smb", "hml"]]


def parse_shiller_xls(content: bytes) -> pd.DataFrame:
    """Shiller ie_data.xls 'Data' sheet: Date as YYYY.MM, P, D, E and a CAPE column."""
    raw = pd.read_excel(io.BytesIO(content), sheet_name="Data", header=None, engine="xlrd")
    header_row = next(i for i in range(min(len(raw), 30)) if str(raw.iat[i, 0]).strip() == "Date")
    headers = [str(h).strip() for h in raw.iloc[header_row]]
    data = raw.iloc[header_row + 1:].copy()
    data.columns = headers

    def col(predicate: Callable[[str], bool]) -> str:
        return next(h for h in headers if predicate(h))

    # Prefer the plain "CAPE" column over variants such as "TR CAPE"
    cape_col = next((h for h in headers if h.upper() == "CAPE"), None) or col(lambda h: "CAPE" in h.upper())
    out = pd.DataFrame({
        "date": pd.to_numeric(data["Date"], errors="coerce"),
        "price": pd.to_numeric(data[col(lambda h: h == "P")], errors="coerce"),
        "dividend": pd.to_numeric(data[col(lambda h: h == "D")], errors="coerce"),
        "earnings": pd.to_numeric(data[col(lambda h: h == "E")], errors="coerce"),
        "cape": pd.to_numeric(data[cape_col], errors="coerce"),
    }).dropna(subset=["date", "price"])
    # 1871.01 is January; 1871.1 is October (stored as a float)
    years = np.floor(out["date"]).astype(int)
    months = np.rint((out["date"] - years) * 100).astype(int)
    out.index = pd.to_datetime({"year": years, "month": months, "day": 1}) + pd.offsets.MonthEnd(0)
    out.index.name = "date"
    out["earnings_yield"] = out["earnings"] / out["price"]
    out["dividend_yield"] = out["dividend"] / out["price"]
    return out[["cape", "earnings_yield", "dividend_yield"]]


# =============================================================================
# Manual inputs (survey returns and equity valuations)
# =============================================================================

def write_manual_inputs_template(path: str | Path, universe_map: Dict[str, Dict[str, Any]]) -> Path:
    """Create data/manual_inputs.xlsx with one row per asset class. Refuses to overwrite."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} already exists; it holds your inputs, so it is not overwritten.")
    path.parent.mkdir(parents=True, exist_ok=True)
    equity = [a for a, m in universe_map.items() if m["category"] == EQUITY_CATEGORY]
    survey = pd.DataFrame({
        "survey_date": pd.NaT,
        "asset_class": list(universe_map),
        "survey_expected_return (%)": np.nan,
    })
    valuations = pd.DataFrame({
        "date": pd.NaT,
        "asset_class": equity,
        "cape_ratio": np.nan,
        "earnings_yield (%)": np.nan,
        "dividend_yield (%)": np.nan,
    })
    instructions = pd.DataFrame({"how to fill this workbook": [
        "survey_cma: the long-term expected return for each asset class from a published survey of capital "
        "market assumptions (for example the free annual Horizon Actuarial survey), in percent, with the "
        "survey's publication date. Required for the six equity classes; the others are optional.",
        "Add a new row for each new survey; keep the old rows. The pipeline uses the latest one before the as-of "
        "date and lowers its weight as it ages.",
        "equity_valuations: optional. CAPE, earnings yield and dividend yield for each equity class (for example "
        "from MSCI or iShares fact sheets). Blank rows use the S&P 500 values from Robert Shiller's data as a proxy.",
        "Values are carried forward until the next row for the same asset class.",
    ]})
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        instructions.to_excel(writer, sheet_name="INSTRUCTIONS", index=False)
        survey.to_excel(writer, sheet_name="survey_cma", index=False)
        valuations.to_excel(writer, sheet_name="equity_valuations", index=False)
    logger.info("Wrote manual inputs template to %s", path)
    return path


def _read_manual_inputs(path: Path) -> Dict[str, pd.DataFrame]:
    sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    for sheet in sheets.values():
        sheet.columns = [str(c).split(" (")[0].strip() for c in sheet.columns]
    return sheets


# =============================================================================
# Assembly
# =============================================================================

def _month_end_last(series: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    return series.resample("ME").last()


def _first_of_month_to_month_end(series: pd.Series) -> pd.Series:
    out = series.copy()
    out.index = out.index + pd.offsets.MonthEnd(0)
    return out


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's relative strength index."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    return 100 - 100 / (1 + gain / loss)


def _latest_common_month_end(frames: Dict[str, pd.DataFrame]) -> pd.Timestamp:
    """The last month-end that every time series reaches (survey excluded)."""
    ends = []
    for name, df in frames.items():
        if name == "df_survey_cma_raw":
            continue
        dates = df.index.get_level_values("date") if isinstance(df.index, pd.MultiIndex) else df.index
        last = dates.max()
        month_end = last + pd.offsets.MonthEnd(0)
        ends.append(month_end if last == month_end else last - pd.offsets.MonthEnd(1))
    return min(ends)


def build_free_pipeline_data(
    universe_map: Dict[str, Dict[str, Any]],
    start_date: str = "1990-01-01",
    manual_inputs_path: str | Path = MANUAL_INPUTS_PATH,
    sources: Optional[FreeDataSources] = None,
) -> Dict[str, Any]:
    """
    Download and assemble the 10 pipeline input dataframes from free sources.

    Returns a dict with the 10 dataframes plus 'suggested_as_of_date': the last
    month-end every source covers (use it as AS_OF_DATE). Raises if the manual
    inputs workbook lacks survey returns for the equity asset classes.
    """
    sources = sources or FreeDataSources()
    manual_inputs_path = Path(manual_inputs_path)
    if not manual_inputs_path.exists():
        write_manual_inputs_template(manual_inputs_path, universe_map)
        raise FileNotFoundError(
            f"Created {manual_inputs_path}. Fill in its survey_cma sheet (see INSTRUCTIONS), then run again."
        )
    manual = _read_manual_inputs(manual_inputs_path)
    fred = {key: sources.fred(series_id) for key, series_id in FRED_SERIES.items()}

    # ---- Macro (monthly, decimals) --------------------------------------------------------
    gdp = fred["real_gdp_growth"] / 100
    gdp.index = gdp.index + pd.offsets.QuarterEnd(0)          # quarter start -> quarter end
    cpi = _first_of_month_to_month_end(fred["cpi_sa"])
    payrolls = _first_of_month_to_month_end(fred["payrolls"])
    fed_funds = _first_of_month_to_month_end(fred["fed_funds"]) / 100
    fci = _month_end_last(fred["fci"])
    macro = pd.DataFrame({
        "real_gdp_growth_rev": gdp,
        "nonfarm_payrolls_mom": payrolls.diff() * 1000,
        "cpi_yoy": cpi.pct_change(12),
        "cpi_mom": cpi.pct_change(),
        "cpi_core_yoy": _first_of_month_to_month_end(fred["cpi_core_sa"]).pct_change(12),
        "brent_crude_usd": _month_end_last(fred["brent"]),
        "fed_funds_rate": fed_funds,
        "fed_funds_3m_change": fed_funds.diff(3),
        "financial_conditions_index": fci,
        "financial_conditions_mom": fci.diff(),
    }).sort_index()
    macro["real_gdp_growth_rev"] = macro["real_gdp_growth_rev"].ffill(limit=3)
    macro = macro.loc[pd.Timestamp(start_date):]
    macro = macro.loc[macro.notna().all(axis=1).idxmax():]     # first month with every indicator
    macro = macro.loc[: macro.dropna(how="all").index.max()]
    macro.index.name = "date"

    cpi_index = _first_of_month_to_month_end(fred["cpi_nsa"]).to_frame("cpi_index_level").loc[start_date:]
    cpi_index.index.name = "date"

    rates = pd.DataFrame({
        "ust_3m_yield": _month_end_last(fred["ust_3m"]),
        "ust_2y_yield": _month_end_last(fred["ust_2y"]),
        "ust_10y_yield": _month_end_last(fred["ust_10y"]),
        "ust_30y_yield": _month_end_last(fred["ust_30y"]),
        "ig_oas": _month_end_last(fred["ig_oas"]),
        "hy_oas": _month_end_last(fred["hy_oas"]),
        "em_spread": _month_end_last(fred["em_oas"]),
    }).loc[start_date:] / 100
    rates.index.name = "date"

    # ---- Daily benchmark and factors ------------------------------------------------------
    eq = sources.yahoo(BENCHMARK_EQUITY_SYMBOL, start_date)["Adj Close"]
    bd = sources.yahoo(BENCHMARK_BOND_SYMBOL, start_date)["Adj Close"]
    bench = pd.concat({"equity_leg_total_return_index": eq, "bond_leg_total_return_index": bd}, axis=1).dropna()
    bench["w_equity"] = 0.60
    bench["w_bond"] = 0.40
    bench.index.name = "date"

    factors = (sources.french_daily() / 100).loc[start_date:]
    factors.index.name = "date"

    # ---- Per-asset panels (monthly) -------------------------------------------------------
    tickers = [m["ticker"] for m in universe_map.values()]
    asset_by_ticker = {m["ticker"]: a for a, m in universe_map.items()}
    category_by_ticker = {m["ticker"]: m["category"] for m in universe_map.values()}

    monthly: Dict[str, Dict[str, pd.Series]] = {f: {} for f in ("tri", "open", "high", "low", "close", "volume", "rsi")}
    for ticker in tickers:
        px = sources.yahoo(YAHOO_SYMBOLS.get(ticker, ticker), start_date)
        m = px.resample("ME")
        monthly["tri"][ticker] = m["Adj Close"].last()
        monthly["open"][ticker] = m["Open"].first()
        monthly["high"][ticker] = m["High"].max()
        monthly["low"][ticker] = m["Low"].min()
        monthly["close"][ticker] = m["Close"].last()
        monthly["volume"][ticker] = m["Volume"].sum(min_count=1)
        monthly["rsi"][ticker] = _rsi(px["Adj Close"]).resample("ME").last()
    wide = {k: pd.DataFrame(v).reindex(columns=tickers) for k, v in monthly.items()}
    dates = wide["tri"].index
    wide["momentum"] = wide["tri"] / wide["tri"].shift(12) - 1

    # Equity valuations: manual rows where given, else Shiller S&P 500 as a proxy
    shiller = sources.shiller().reindex(dates, method="ffill")
    val = {k: pd.DataFrame(np.nan, index=dates, columns=tickers) for k in ("cape", "ey", "dy")}
    manual_val = manual.get("equity_valuations", pd.DataFrame()).dropna(subset=["date", "asset_class"])
    proxied: List[str] = []
    for ticker in tickers:
        if category_by_ticker[ticker] != EQUITY_CATEGORY:
            continue
        rows = manual_val[manual_val["asset_class"] == asset_by_ticker[ticker]]
        if rows.empty:
            proxied.append(asset_by_ticker[ticker])
            val["cape"][ticker], val["ey"][ticker], val["dy"][ticker] = (
                shiller["cape"], shiller["earnings_yield"], shiller["dividend_yield"])
            continue
        rows = rows.assign(date=pd.to_datetime(rows["date"]) + pd.offsets.MonthEnd(0)).set_index("date").sort_index()
        for key, col, scale in (("cape", "cape_ratio", 1.0), ("ey", "earnings_yield", 0.01), ("dy", "dividend_yield", 0.01)):
            val[key][ticker] = (pd.to_numeric(rows[col], errors="coerce") * scale).reindex(dates, method="ffill")
    if proxied:
        logger.warning("No manual valuations for %s: using S&P 500 values from Shiller's data as a proxy.", proxied)

    index = pd.MultiIndex.from_tuples(
        [(d, t, category_by_ticker[t]) for d in dates for t in tickers],
        names=["date", "ticker", "investment_universe"],
    )

    def stack(df: Optional[pd.DataFrame]) -> np.ndarray:
        if df is None:
            return np.full(len(index), np.nan)
        return df.reindex(index=dates, columns=tickers).to_numpy(dtype="float64").reshape(-1)

    total_return = pd.DataFrame({"total_return_index": stack(wide["tri"])}, index=index)
    ohlcv = pd.DataFrame({f: stack(wide[f]) for f in ("open", "high", "low", "close", "volume")}, index=index)
    fundamentals = pd.DataFrame({
        "cape_ratio": stack(val["cape"]),
        "pe_trailing": stack(1 / val["ey"]),
        "earnings_yield": stack(val["ey"]),
        "dividend_yield": stack(val["dy"]),
        "buyback_yield": stack(None),
        "market_cap_usd": stack(None),
        "earnings_growth_forecast": stack(None),
        "valuation_change_assumption": stack(None),
    }, index=index)
    signals = pd.DataFrame({
        "rsi_14d": stack(wide["rsi"]),
        "momentum_12m": stack(wide["momentum"]),
        "market_breadth_raw": stack(None),
        "net_fund_flows": stack(None),
        "positioning_score_raw": stack(None),
    }, index=index)

    # ---- Survey expected returns (manual) --------------------------------------------------
    survey = manual.get("survey_cma", pd.DataFrame()).dropna(subset=["survey_date", "asset_class", "survey_expected_return"])
    unknown = sorted(set(survey["asset_class"]) - set(universe_map))
    if unknown:
        raise ValueError(f"{manual_inputs_path} survey_cma has unknown asset classes: {unknown}")
    missing = [a for a, m in universe_map.items() if m["category"] == EQUITY_CATEGORY and a not in set(survey["asset_class"])]
    if missing:
        raise ValueError(
            f"{manual_inputs_path} survey_cma needs a survey_date and survey_expected_return for: {missing}"
        )
    survey = pd.DataFrame({
        "asset_class": survey["asset_class"].astype("object").to_numpy(),
        "survey_expected_return": pd.to_numeric(survey["survey_expected_return"]).to_numpy(dtype="float64") / 100,
    }, index=pd.DatetimeIndex(pd.to_datetime(survey["survey_date"]) + pd.offsets.MonthEnd(0), name="date")).sort_index()

    frames: Dict[str, Any] = {
        "df_macro_raw": macro.astype("float64"),
        "df_benchmark_factors_raw": factors.astype("float64"),
        "df_benchmark_60_40_raw": bench.astype("float64"),
        "df_fixed_income_curves_spreads_raw": rates.astype("float64"),
        "df_cpi_index_raw": cpi_index.astype("float64"),
        "df_survey_cma_raw": survey,
        "df_total_return_raw": total_return,
        "df_ohlcv_raw": ohlcv,
        "df_fundamentals_raw": fundamentals,
        "df_signals_raw": signals,
    }
    frames["suggested_as_of_date"] = _latest_common_month_end(
        {k: v for k, v in frames.items() if isinstance(v, pd.DataFrame)}
    ).strftime("%Y-%m-%d")
    logger.info("Free data assembled; every source covers through %s.", frames["suggested_as_of_date"])
    return frames
