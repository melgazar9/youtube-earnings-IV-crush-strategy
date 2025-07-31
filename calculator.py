import yfinance as yf
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
import numpy as np
import argparse
import warnings
import pandas as pd
import os
import requests
import plotly.express as px
from dateutil.relativedelta import relativedelta

warnings.filterwarnings("ignore", message="Not enough unique days to interpolate for ticker")

### ------ Globals ------ ###

MIN_AVG_30D_DOLLAR_VOLUME = 10_000_000
MIN_AVG_30D_SHARE_VOLUME = 1_500_500
MIN_IV30_RV30 = 1.35
MAX_TS_SLOPE_0_45 = -0.0050
MIN_SHARE_PRICE = 15
EARNINGS_LOOKBACK_DAYS_FOR_AGG = 365 * 3
PLOT_LOC = f"{os.path.expanduser("~/")}tmp_plots/"


def filter_dates(dates):
    today = datetime.today().date()
    cutoff_date = today + timedelta(days=45)

    sorted_dates = sorted(datetime.strptime(date, "%Y-%m-%d").date() for date in dates)

    arr = []
    for i, date in enumerate(sorted_dates):
        if date >= cutoff_date:
            arr = [d.strftime("%Y-%m-%d") for d in sorted_dates[: i + 1]]
            break

    if len(arr) > 0:
        if arr[0] == today.strftime("%Y-%m-%d"):
            return arr[1:]
        return arr

    raise ValueError("No date 45 days or more in the future found.")


def yang_zhang(price_data, window=30, trading_periods=252, return_last_only=True):
    log_ho = (price_data["High"] / price_data["Open"]).apply(np.log)
    log_lo = (price_data["Low"] / price_data["Open"]).apply(np.log)
    log_co = (price_data["Close"] / price_data["Open"]).apply(np.log)

    log_oc = (price_data["Open"] / price_data["Close"].shift(1)).apply(np.log)
    log_oc_sq = log_oc ** 2

    log_cc = (price_data["Close"] / price_data["Close"].shift(1)).apply(np.log)
    log_cc_sq = log_cc ** 2

    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

    close_vol = log_cc_sq.rolling(window=window, center=False).sum() * (1.0 / (window - 1.0))

    open_vol = log_oc_sq.rolling(window=window, center=False).sum() * (1.0 / (window - 1.0))

    window_rs = rs.rolling(window=window, center=False).sum() * (1.0 / (window - 1.0))

    k = 0.3333 / (1.3333 + ((window + 1) / (window - 1)))
    result = (open_vol + k * close_vol + (1 - k) * window_rs).apply(np.sqrt) * np.sqrt(trading_periods)

    if return_last_only:
        return result.iloc[-1]
    else:
        return result.dropna()


def build_term_structure(days, ivs):
    days = np.array(days)
    ivs = np.array(ivs)

    # Sort by days
    sort_idx = days.argsort()
    days = days[sort_idx]
    ivs = ivs[sort_idx]

    _, unique_idx = np.unique(days, return_index=True)
    days = days[sorted(unique_idx)]
    ivs = ivs[sorted(unique_idx)]

    if len(days) < 2:
        warnings.warn(f"Not enough unique days to interpolate for ticker {ticker}.")
        return

    spline = interp1d(days, ivs, kind="linear", fill_value="extrapolate")

    def term_spline(dte):
        if dte < days[0]:
            return ivs[0]
        elif dte > days[-1]:
            return ivs[-1]
        else:
            return float(spline(dte))

    return term_spline


def get_current_price(df_price_history_3mo):
    return df_price_history_3mo["Close"].iloc[-1]


def calc_kelly_bet(p_win: float = 0.66, odds_decimal: float = 1.66, current_bankroll: float = 10000,
                   pct_kelly=0.10) -> float:
    """
    Calculates the Kelly Criterion optimal bet amount.

    The Kelly Criterion is a formula used to determine the optimal size of a series
    of bets to maximize the long-term growth rate of a bankroll.

    Args:
        p_win: The estimated probability of winning the bet (p),
                                a float between 0 and 1.
        odds_decimal: The decimal odds (b), where a successful $1 bet returns $b.
                      For example, if odds are 2:1, odds_decimal is 3.0.
                      If odds are 1:1, odds_decimal is 2.0.
                      This is (payout / stake) + 1.
        current_bankroll: The total amount of money available to bet (B).

    Returns:
        The calculated optimal bet amount. Returns 0 if the bet is not favorable
        (i.e., the calculated fraction is negative or zero), or if inputs are invalid.
    """
    if not (0 <= p_win <= 1):
        raise ValueError("Probability of winning must be between 0 and 1.")
    if odds_decimal <= 1.0:  # Odds must be greater than 1.0 (e.g., 1.01 for a tiny profit)
        raise ValueError("Decimal odds must be greater than 1.0 (e.g., 1.01 for a winning bet).")
    if current_bankroll <= 0:
        raise ValueError("Current bankroll must be a positive number.")

    b_kelly = odds_decimal - 1.0

    if b_kelly <= 0:  # Should be caught by odds_decimal check, but as a safeguard
        return 0.0

    kelly_fraction = p_win - ((1 - p_win) / b_kelly)

    if kelly_fraction <= 0:
        return 0.0

    bet_amount = kelly_fraction * current_bankroll
    bet_amount = bet_amount * pct_kelly
    return round(bet_amount, 2)


def get_all_usa_tickers(use_yf_db=True, earnings_date=datetime.today().strftime("%Y-%m-%d")):
    ### FMP ###

    try:
        fmp_apikey = os.getenv("FMP_API_KEY")
        fmp_url = (
            f"https://financialmodelingprep.com/api/v3/earning_calendar?from={earnings_date}&to={earnings_date}&apikey={fmp_apikey}"
        )
        fmp_response = requests.get(fmp_url)
        df_fmp = pd.DataFrame(fmp_response.json())
        df_fmp_usa = df_fmp[df_fmp["symbol"].str.fullmatch(r"[A-Z]{1,4}") & ~df_fmp["symbol"].str.contains(r"[.-]")]

        fmp_usa_symbols = sorted(df_fmp_usa["symbol"].unique().tolist())
    except Exception:
        print("No FMP API Key found. Only using NASDAQ")
        fmp_usa_symbols = []

    ### NASDAQ ###

    nasdaq_url = f"https://api.nasdaq.com/api/calendar/earnings?date={earnings_date}"
    nasdaq_headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    nasdaq_response = requests.get(nasdaq_url, headers=nasdaq_headers)
    nasdaq_calendar = nasdaq_response.json().get("data").get("rows")
    df_nasdaq = pd.DataFrame(nasdaq_calendar)
    df_nasdaq = df_nasdaq[df_nasdaq["symbol"].str.fullmatch(r"[A-Z]{1,4}") & ~df_nasdaq["symbol"].str.contains(r"[.-]")]

    nasdaq_tickers = sorted(df_nasdaq["symbol"].unique().tolist())

    all_usa_earnings_tickers_today = sorted(list(set(fmp_usa_symbols + nasdaq_tickers)))

    return all_usa_earnings_tickers_today


def calc_prev_earnings_stats(df_history, ticker_obj, ticker, plot_loc=PLOT_LOC):
    df_history = df_history.copy()
    if "Date" not in df_history.columns and df_history.index.name == "Date":
        df_history = df_history.reset_index()
    df_history["Date"] = df_history["Date"].dt.date
    df_history = df_history.sort_values("Date")

    n_tries = 3
    i = 0
    while i < n_tries:
        df_earnings_dates = ticker_obj.earnings_dates
        if df_earnings_dates is not None and not df_earnings_dates.empty:
            break
        i += 1

    if df_earnings_dates is None:
        return 0, 0, 0, 0, 0, 0, None

    df_earnings_dates = df_earnings_dates.reset_index()
    df_earnings_dates = df_earnings_dates[df_earnings_dates["Event Type"] == "Earnings"].copy()
    df_earnings_dates["Date"] = df_earnings_dates["Earnings Date"].dt.date

    def classify_release(dt):
        hour = dt.hour
        if hour < 9:
            return "pre-market"
        elif hour >= 9:
            return "post-market"

    df_earnings_dates["release_timing"] = df_earnings_dates["Earnings Date"].apply(classify_release)
    df_earnings = df_earnings_dates.merge(df_history, on="Date", how="left", suffixes=('', '_earnings'))
    df_earnings["next_date"] = df_earnings["Date"] + pd.Timedelta(days=1)
    df_next = df_history.rename(columns=lambda c: f"{c}_next" if c != "Date" else "next_date")
    df_flat = df_earnings.merge(df_next, on="next_date", how="left")
    df_flat["prev_close"] = df_flat["Close"].shift(1)
    df_flat["pre_market_move"] = (df_flat["Open"] - df_flat["prev_close"]) / df_flat["prev_close"]
    df_flat["post_market_move"] = (df_flat["Open_next"] - df_flat["Close"]) / df_flat["Close"]

    df_flat["earnings_move"] = df_flat.apply(
        lambda row: row["pre_market_move"] if row["release_timing"] == "pre-market"
        else row["post_market_move"] if row["release_timing"] == "post-market"
        else None,
        axis=1
    )

    if plot_loc and df_flat.shape[0]:
        df_flat["text"] = (df_flat["earnings_move"] * 100).round(2).astype(str) + "%"
        p = px.bar(
            x=df_flat["Date"],
            y=df_flat["earnings_move"].round(3),
            color=df_flat.index.astype(str),
            text=df_flat["text"],
            title="Earnings % Move",
        )
        p.update_traces(textangle=0)
        # p.show()

        full_path = os.path.join(plot_loc, f"{ticker}_{df_flat["Date"].iloc[0].strftime("%Y-%m-%d")}.html")
        os.makedirs(plot_loc, exist_ok=True)
        p.write_html(full_path)
        print(f"Saved plot for ticker {ticker} here: {full_path}")

    avg_abs_pct_move = round(abs(df_flat["earnings_move"]).mean(), 3)
    prev_earnings_std = round(abs(df_flat["earnings_move"]).std(ddof=1), 3)
    median_abs_pct_move = round(abs(df_flat["earnings_move"]).median(), 3)
    min_abs_pct_move = round(abs(df_flat["earnings_move"]).min(), 3)
    max_abs_pct_move = round(abs(df_flat["earnings_move"]).max(), 3)
    earnings_release_timing_mode = df_flat["release_timing"].mode()
    release_time = earnings_release_timing_mode.iloc[0] if not earnings_release_timing_mode.empty else "unknown"
    prev_earnings_values = df_flat["earnings_move"].dropna().values

    if prev_earnings_std < 0.001:
        prev_earnings_std = 0.001  # avoid division by 0 or overly tight thresholds

    return avg_abs_pct_move, median_abs_pct_move, min_abs_pct_move, max_abs_pct_move, prev_earnings_std, release_time, prev_earnings_values


def compute_recommendation(
        ticker,
        min_avg_30d_dollar_volume=MIN_AVG_30D_DOLLAR_VOLUME,
        min_avg_30d_share_volume=MIN_AVG_30D_SHARE_VOLUME,
        min_iv30_rv30=MIN_IV30_RV30,
        max_ts_slope_0_45=MAX_TS_SLOPE_0_45,
        plot_loc=PLOT_LOC,
):
    ticker = ticker.strip().upper()
    if not ticker:
        return "No stock symbol provided."

    try:
        stock = yf.Ticker(ticker)
        n_tries = 3
        i = 0
        while i < n_tries:
            exp_dates = list(stock.options)
            if exp_dates:
                break
            i += 1
        if len(exp_dates) == 0:
            raise KeyError(f"No options data found for ticker {ticker}")
    except KeyError:
        return f"Error: No options found for stock symbol '{ticker}'."

    try:
        exp_dates = filter_dates(exp_dates)
    except Exception:
        return "Error: Not enough option data."

    options_chains = {}
    for exp_date in exp_dates:
        n_tries = 3
        i = 0
        while i < n_tries:
            chain = stock.option_chain(exp_date)
            options_chains[exp_date] = chain
            if chain is not None and len(chain):
                break
            i += 1

    n_tries = 3
    i = 0
    while i < n_tries:
        df_history = stock.history(
            start=(datetime.today() - timedelta(days=EARNINGS_LOOKBACK_DAYS_FOR_AGG)).strftime("%Y-%m-%d"))
        if df_history is not None and not df_history.empty:
            break
        i += 1

    # df_price_history_3mo = stock.history(period="3mo")

    df_price_history_3mo = df_history[
        df_history.index >= (pd.Timestamp.now(df_history.index.tz) - relativedelta(months=3))]
    # df_price_history_3mo = df_history[df_history.index >= (datetime.now(pytz.timezone("America/New_York")) - relativedelta(months=3))]
    df_price_history_3mo = df_price_history_3mo.sort_index()
    df_price_history_3mo["dollar_volume"] = df_price_history_3mo["Volume"] * df_price_history_3mo["Close"]

    try:
        underlying_price = get_current_price(df_price_history_3mo)
        if underlying_price is None:
            raise ValueError("No market price found.")
    except Exception:
        return "Error: Unable to retrieve underlying stock price."

    atm_iv = {}
    straddle = None
    i = 0
    for exp_date, chain in options_chains.items():
        calls = chain.calls
        puts = chain.puts

        if calls is None or puts is None or calls.empty or puts.empty:
            continue

        call_diffs = (calls["strike"] - underlying_price).abs()
        call_idx = call_diffs.idxmin()
        call_iv = calls.loc[call_idx, "impliedVolatility"]

        put_diffs = (puts["strike"] - underlying_price).abs()
        put_idx = put_diffs.idxmin()
        put_iv = puts.loc[put_idx, "impliedVolatility"]

        atm_iv_value = (call_iv + put_iv) / 2.0
        atm_iv[exp_date] = atm_iv_value

        if i == 0:
            call_bid = calls.loc[call_idx, "bid"]
            call_ask = calls.loc[call_idx, "ask"]
            put_bid = puts.loc[put_idx, "bid"]
            put_ask = puts.loc[put_idx, "ask"]

            if call_bid is not None and call_ask is not None:
                call_mid = (call_bid + call_ask) / 2.0
            else:
                call_mid = None

            if put_bid is not None and put_ask is not None:
                put_mid = (put_bid + put_ask) / 2.0
            else:
                put_mid = None

            if call_mid is not None and put_mid is not None and call_mid != 0 and put_mid != 0:
                straddle = call_mid + put_mid
            else:
                try:
                    if call_idx + 1 < len(calls) and put_idx + 1 < len(puts):
                        warnings.warn(f"For ticker {ticker} straddle is either 0 or None from available bid/ask spread... using nearest term strikes.")
                        straddle = calls.iloc[call_idx + 1]["lastPrice"] + puts.iloc[put_idx + 1]["lastPrice"]
                    if not straddle:
                        warnings.warn(f"For ticker {ticker} straddle is either 0 or None from available bid/ask spread... using lastPrice.")
                        straddle = calls.iloc[call_idx]["lastPrice"] + puts.iloc[call_idx]["lastPrice"]
                except IndexError:
                    warnings.warn(f"For ticker {ticker}, call_idx {call_idx} is out of bounds in calls/puts.")
                    return None
        i += 1

    if not atm_iv:
        return "Error: Could not determine ATM IV for any expiration dates."

    today = datetime.today().date()
    dtes = []
    ivs = []
    for exp_date, iv in atm_iv.items():
        exp_date_obj = datetime.strptime(exp_date, "%Y-%m-%d").date()
        days_to_expiry = (exp_date_obj - today).days
        dtes.append(days_to_expiry)
        ivs.append(iv)

    term_spline = build_term_structure(dtes, ivs)
    if not term_spline:
        return

    ts_slope_0_45 = (term_spline(45) - term_spline(dtes[0])) / (45 - dtes[0])

    iv30_rv30 = term_spline(30) / yang_zhang(df_price_history_3mo)

    rolling_share_volume = df_price_history_3mo["Volume"].rolling(30).mean().dropna()
    rolling_dollar_volume = df_price_history_3mo["dollar_volume"].rolling(30).mean().dropna()

    if rolling_share_volume.empty:
        avg_share_volume = 0
    else:
        avg_share_volume = rolling_share_volume.iloc[-1]

    if rolling_dollar_volume.empty:
        avg_dollar_volume = 0
    else:
        avg_dollar_volume = rolling_dollar_volume.iloc[-1]

    expected_move_straddle = (straddle / underlying_price).round(3) if straddle else None

    (
        prev_earnings_avg_abs_pct_move, prev_earnings_median_abs_pct_move, prev_earnings_min_abs_pct_move,
        prev_earnings_max_abs_pct_move, prev_earnings_std, earnings_release_time, prev_earnings_values
     ) = calc_prev_earnings_stats(df_history.reset_index(), stock, ticker, plot_loc=plot_loc)

    if prev_earnings_values is None or not len(prev_earnings_values):
        prev_earnings_values = []

    result_summary = {
        "avg_30d_dollar_volume": round(avg_dollar_volume, 3),
        "avg_30d_dollar_volume_pass": avg_dollar_volume >= min_avg_30d_dollar_volume,
        "avg_30d_share_volume": round(avg_share_volume, 3),
        "avg_30d_share_volume_pass": avg_share_volume >= min_avg_30d_share_volume,
        "iv30_rv30": round(iv30_rv30, 3),
        "iv30_rv30_pass": iv30_rv30 >= min_iv30_rv30,
        "ts_slope_0_45": ts_slope_0_45,
        "ts_slope_0_45_pass": ts_slope_0_45 <= max_ts_slope_0_45,
        "underlying_price": underlying_price,
        "call_spread": (call_bid, call_ask),
        "put_spread": (put_bid, put_ask),
        "expected_move_straddle": (expected_move_straddle * 100).round(3).astype(str) + "%",
        "straddle_pct_move_ge_hist_pct_move_pass": expected_move_straddle >= prev_earnings_avg_abs_pct_move,
        "prev_earnings_avg_abs_pct_move": str(round(prev_earnings_avg_abs_pct_move * 100, 3)) + "%",
        "prev_earnings_median_abs_pct_move": str(round(prev_earnings_median_abs_pct_move * 100, 3)) + "%",
        "prev_earnings_min_abs_pct_move": str(round(prev_earnings_min_abs_pct_move * 100, 3)) + "%",
        "prev_earnings_max_abs_pct_move": str(round(prev_earnings_max_abs_pct_move * 100, 3)) + "%",
        "prev_earnings_values": prev_earnings_values,
        "earnings_release_time": earnings_release_time
    }

    if (
            result_summary["avg_30d_dollar_volume_pass"]
            and result_summary["iv30_rv30_pass"]
            and result_summary["ts_slope_0_45_pass"]
            and result_summary["avg_30d_share_volume_pass"]
    ):
        original_suggestion = "Recommended"
    elif result_summary["ts_slope_0_45_pass"] and (
            (result_summary["avg_30d_dollar_volume_pass"] and not result_summary["iv30_rv30_pass"])
            or (result_summary["iv30_rv30_pass"] and not result_summary["avg_30d_dollar_volume_pass"])
    ):
        original_suggestion = "Consider"
    else:
        original_suggestion = "Avoid"

    if (
            result_summary["avg_30d_dollar_volume_pass"]
            and result_summary["iv30_rv30_pass"]
            and result_summary["ts_slope_0_45_pass"]
            and result_summary["avg_30d_share_volume_pass"]
            and result_summary["underlying_price"] >= MIN_SHARE_PRICE
            and result_summary["straddle_pct_move_ge_hist_pct_move_pass"]
            and expected_move_straddle > prev_earnings_min_abs_pct_move  # safety filter - data quality check
    ):
        improved_suggestion = "Highly Recommended"
    elif (
            result_summary["avg_30d_dollar_volume_pass"]
            and result_summary["iv30_rv30_pass"]
            and result_summary["ts_slope_0_45_pass"]
            and result_summary["avg_30d_share_volume_pass"]
            and result_summary["underlying_price"] >= MIN_SHARE_PRICE
            and prev_earnings_avg_abs_pct_move - expected_move_straddle <= 0.75 * prev_earnings_std  # Avg move - Straddle is within 0.75 std deviations
            and expected_move_straddle > prev_earnings_min_abs_pct_move  # Safety filter - data quality check
    ):
        improved_suggestion = "Slightly Recommended"
    elif (
            result_summary["avg_30d_dollar_volume_pass"]
            and result_summary["iv30_rv30_pass"]
            and result_summary["ts_slope_0_45_pass"]
            and result_summary["avg_30d_share_volume_pass"]
            and result_summary["underlying_price"] >= MIN_SHARE_PRICE
            and prev_earnings_avg_abs_pct_move - expected_move_straddle <= 0.50 * prev_earnings_std  # Avg move - Straddle is within 0.50 std deviations
            and expected_move_straddle > prev_earnings_min_abs_pct_move  # Safety filter - data quality check
    ):
        improved_suggestion = "Recommended"
    elif result_summary["ts_slope_0_45_pass"] and result_summary["avg_30d_dollar_volume_pass"] and result_summary[
        "iv30_rv30_pass"] and expected_move_straddle * 1.5 < prev_earnings_min_abs_pct_move:
        improved_suggestion = "Consider..."
    elif result_summary["ts_slope_0_45_pass"] and result_summary["avg_30d_dollar_volume_pass"] and result_summary[
        "iv30_rv30_pass"]:
        improved_suggestion = "Slightly Consider..."
    elif result_summary["ts_slope_0_45_pass"] and (
            (result_summary["avg_30d_dollar_volume_pass"] and not result_summary["iv30_rv30_pass"])
            or (result_summary["iv30_rv30_pass"] and not result_summary["avg_30d_dollar_volume_pass"])
    ):
        improved_suggestion = "Eh... Consider, but it's risky!"
    else:
        improved_suggestion = "Avoid"

    edge_score = 0

    # IV to RV ratio
    if iv30_rv30 > 2.0:
        edge_score += 1.0
    elif iv30_rv30 > 1.5:
        edge_score += 0.5

    # Term structure slope
    if ts_slope_0_45 < -0.01:
        edge_score += 0.5

    # Liquidity
    if avg_dollar_volume > 50_000_000:
        edge_score += 0.5

    # Straddle expected pct change >= avg earnings pct change
    if expected_move_straddle >= prev_earnings_avg_abs_pct_move:
        edge_score += 1.0

    if "Recommended" in improved_suggestion:
        if edge_score >= 3.0:
            kelly_multiplier_from_base = 2.0
        elif edge_score >= 2.5:
            kelly_multiplier_from_base = 1.75
        elif edge_score >= 2.0:
            kelly_multiplier_from_base = 1.5
        elif edge_score >= 1.5:
            kelly_multiplier_from_base = 1.25
        elif edge_score >= 1:
            kelly_multiplier_from_base = 1.125
        elif edge_score >= 0.5:
            kelly_multiplier_from_base = 1.0
        elif edge_score == 0:
            kelly_multiplier_from_base = 0.80
    elif "Consider" in improved_suggestion:
        kelly_multiplier_from_base = 0.5
    elif original_suggestion == "Consider":
        kelly_multiplier_from_base = 0.2
    else:
        kelly_multiplier_from_base = 0

    result_summary["improved_suggestion"] = improved_suggestion
    result_summary["original_suggestion"] = original_suggestion
    kelly_bet = calc_kelly_bet()

    kelly_bet = round(kelly_bet * kelly_multiplier_from_base, 2)
    result_summary["kelly_multiplier_from_base"] = kelly_multiplier_from_base
    result_summary["kelly_bet"] = kelly_bet
    return result_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run calculations for given tickers")

    parser.add_argument("--earnings-date", type=str, default=datetime.today().strftime("%Y-%m-%d"),
                        help="Earnings date in YYYY-MM-DD format (default: today)")

    parser.add_argument("--tickers", nargs="+", required=True, help="List of ticker symbols (e.g., NVDA AAPL TSLA)")

    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True,
                        help="Verbose output for displaying all results. Default is True.")

    args = parser.parse_args()
    earnings_date = args.earnings_date
    tickers = args.tickers
    verbose = args.verbose

    if tickers == ["_all"]:
        tickers = get_all_usa_tickers(earnings_date=earnings_date)

    print(f"Scanning {len(tickers)} tickers: \n{tickers}\n")

    for ticker in tickers:
        result = compute_recommendation(ticker)
        is_edge = isinstance(result, dict) and "Recommended" in result.get("improved_suggestion")
        if is_edge:
            print(" *** EDGE FOUND ***\n")

        if verbose or is_edge:
            print(f"ticker: {ticker}")
            if isinstance(result, dict):
                for k, v in result.items():
                    print(f"  {k}: {v}")
            else:
                print(f"  {result}")
            print("---------------")
