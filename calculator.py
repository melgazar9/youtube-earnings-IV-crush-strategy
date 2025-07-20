import yfinance as yf
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
import numpy as np
import threading
import argparse

def filter_dates(dates):
    today = datetime.today().date()
    cutoff_date = today + timedelta(days=45)
    
    sorted_dates = sorted(datetime.strptime(date, "%Y-%m-%d").date() for date in dates)

    arr = []
    for i, date in enumerate(sorted_dates):
        if date >= cutoff_date:
            arr = [d.strftime("%Y-%m-%d") for d in sorted_dates[:i+1]]  
            break
    
    if len(arr) > 0:
        if arr[0] == today.strftime("%Y-%m-%d"):
            return arr[1:]
        return arr

    raise ValueError("No date 45 days or more in the future found.")


def yang_zhang(price_data, window=30, trading_periods=252, return_last_only=True):
    log_ho = (price_data['High'] / price_data['Open']).apply(np.log)
    log_lo = (price_data['Low'] / price_data['Open']).apply(np.log)
    log_co = (price_data['Close'] / price_data['Open']).apply(np.log)
    
    log_oc = (price_data['Open'] / price_data['Close'].shift(1)).apply(np.log)
    log_oc_sq = log_oc**2
    
    log_cc = (price_data['Close'] / price_data['Close'].shift(1)).apply(np.log)
    log_cc_sq = log_cc**2
    
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    
    close_vol = log_cc_sq.rolling(
        window=window,
        center=False
    ).sum() * (1.0 / (window - 1.0))

    open_vol = log_oc_sq.rolling(
        window=window,
        center=False
    ).sum() * (1.0 / (window - 1.0))

    window_rs = rs.rolling(
        window=window,
        center=False
    ).sum() * (1.0 / (window - 1.0))

    k = 0.3333 / (1.3333 + ((window + 1) / (window - 1)) )
    result = (open_vol + k * close_vol + (1 - k) * window_rs).apply(np.sqrt) * np.sqrt(trading_periods)

    if return_last_only:
        return result.iloc[-1]
    else:
        return result.dropna()

def build_term_structure(days, ivs):
    days = np.array(days)
    ivs = np.array(ivs)

    sort_idx = days.argsort()
    days = days[sort_idx]
    ivs = ivs[sort_idx]

    spline = interp1d(days, ivs, kind='linear', fill_value="extrapolate")

    def term_spline(dte):
        if dte < days[0]:  
            return ivs[0]
        elif dte > days[-1]:
            return ivs[-1]
        else:  
            return float(spline(dte))
    return term_spline

def get_current_price(df_price_history):
    return df_price_history['Close'].iloc[-1]


def calc_kelly_bet(
    p_win: float = 0.66,
    odds_decimal: float = 1.66,
    current_bankroll: float = 10000,
    pct_kelly=0.10
) -> float:
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
    if odds_decimal <= 1.0: # Odds must be greater than 1.0 (e.g., 1.01 for a tiny profit)
        raise ValueError("Decimal odds must be greater than 1.0 (e.g., 1.01 for a winning bet).")
    if current_bankroll <= 0:
        raise ValueError("Current bankroll must be a positive number.")

    b_kelly = odds_decimal - 1.0

    if b_kelly <= 0: # Should be caught by odds_decimal check, but as a safeguard
        return 0.0

    kelly_fraction = p_win - ((1 - p_win) / b_kelly)

    if kelly_fraction <= 0:
        return 0.0

    bet_amount = kelly_fraction * current_bankroll
    bet_amount = bet_amount * pct_kelly
    return round(bet_amount, 2)

def compute_recommendation(ticker, min_avg_30d_volume=1500000, min_iv30_rv30=2.0, max_ts_slope_0_45=-0.0075):
    try:
        ticker = ticker.strip().upper()
        if not ticker:
            return "No stock symbol provided."
        
        try:
            stock = yf.Ticker(ticker)
            exp_dates = list(stock.options)
            if len(exp_dates) == 0:
                raise KeyError(f"No options data found for ticker {ticker}")
        except KeyError:
            return f"Error: No options found for stock symbol '{ticker}'."
        
        try:
            exp_dates = filter_dates(exp_dates)
        except:
            return "Error: Not enough option data."
        
        options_chains = {}
        for exp_date in exp_dates:
            options_chains[exp_date] = stock.option_chain(exp_date)

        df_price_history = stock.history(period='3mo')
        df_price_history = df_price_history.sort_index()
        
        try:
            underlying_price = get_current_price(df_price_history)
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

            if calls.empty or puts.empty:
                continue

            call_diffs = (calls['strike'] - underlying_price).abs()
            call_idx = call_diffs.idxmin()
            call_iv = calls.loc[call_idx, 'impliedVolatility']

            put_diffs = (puts['strike'] - underlying_price).abs()
            put_idx = put_diffs.idxmin()
            put_iv = puts.loc[put_idx, 'impliedVolatility']

            atm_iv_value = (call_iv + put_iv) / 2.0
            atm_iv[exp_date] = atm_iv_value

            if i == 0:
                call_bid = calls.loc[call_idx, 'bid']
                call_ask = calls.loc[call_idx, 'ask']
                put_bid = puts.loc[put_idx, 'bid']
                put_ask = puts.loc[put_idx, 'ask']
                
                if call_bid is not None and call_ask is not None:
                    call_mid = (call_bid + call_ask) / 2.0
                else:
                    call_mid = None

                if put_bid is not None and put_ask is not None:
                    put_mid = (put_bid + put_ask) / 2.0
                else:
                    put_mid = None

                if call_mid is not None and put_mid is not None:
                    straddle = (call_mid + put_mid)

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
        
        ts_slope_0_45 = (term_spline(45) - term_spline(dtes[0])) / (45-dtes[0])
        
        iv30_rv30 = term_spline(30) / yang_zhang(df_price_history)

        avg_volume = df_price_history['Volume'].rolling(30).mean().dropna().iloc[-1]

        expected_move = str(round(straddle / underlying_price * 100, 2)) + "%" if straddle else None

        result_summary = {
            'avg_volume': round(avg_volume, 3),
            'avg_volume_pass': avg_volume >= min_avg_30d_volume,
            'iv30_rv30': round(iv30_rv30, 3),
            'iv30_rv30_pass': iv30_rv30 >= min_iv30_rv30,
            'ts_slope_0_45': ts_slope_0_45,
            'ts_slope_0_45_pass': ts_slope_0_45 <= max_ts_slope_0_45,
            'expected_move': expected_move
        }
        
        if result_summary['avg_volume_pass'] and result_summary['iv30_rv30_pass'] and result_summary['ts_slope_0_45_pass']:
            suggestion = "Recommended"
        elif result_summary['ts_slope_0_45_pass'] and ((result_summary['avg_volume_pass'] and not result_summary['iv30_rv30_pass']) or (result_summary['iv30_rv30_pass'] and not result_summary['avg_volume_pass'])):
            suggestion = "Consider"
        else:
            suggestion = "Avoid"

        result_summary["suggestion"] = suggestion

        if suggestion == "Recommended":
            kelly_bet = calc_kelly_bet()
        elif suggestion == "Consider":
            kelly_bet = calc_kelly_bet(p_win=0.33)
        else:
            kelly_bet = 0

        result_summary["kelly_bet"] = kelly_bet
        
        # Check that they are in our desired range (see video)
        return result_summary
    
    except Exception as e:
        raise Exception(f'Error occured processing')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run calculations for given tickers")
    parser.add_argument('--tickers', nargs='+', required=True, help='List of ticker symbols (e.g., NVDA AAPL TSLA)')
    
    args = parser.parse_args()
    tickers = args.tickers

    for ticker in tickers:
        result = compute_recommendation(ticker)
        print(f"ticker: {ticker}: {result}")
