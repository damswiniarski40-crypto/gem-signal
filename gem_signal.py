"""
GEM Signal - Global Equities Momentum (Signal Only)

Generates monthly GEM momentum signal without executing trades.
No broker connection. No order execution. Pure signal generator.

Usage: python gem_signal.py
"""

import csv
import datetime
import json
import os
import uuid

import yfinance as yf


# ============================================================================
# CONFIGURATION
# ============================================================================

STRATEGY_POOLS = {
    "Zestaw 1 ETF": {
        "RISK_ON": ["VUAA.L", "EXUS.L"],
        "BOND": "IB01.L",
    },
    "Zestaw 2 ETF": {
        "RISK_ON": ["IWDA.L", "EIMI.L", "CNDX.L"],
        "BOND": "IB01.L",
    },
}

ACTIVE_POOL = "Zestaw 1 ETF"

MOMENTUM_LOOKBACK = 252       # Trading days (12 months)
MIN_SESSIONS = 260            # Minimum sessions required
DOWNLOAD_PERIOD = "460d"      # ~15 months buffer for weekends/holidays
CHURN_THRESHOLD = 0.005       # 0.5% - ignore momentum diff below this
STRICT_GEM_MODE = True        # Use previous month-end momentum for decisions

DECISION_LOG = "decision_history_signal.csv"
HEARTBEAT_FILE = "heartbeat_signal.json"

# --- Derived from active pool (do not edit) ---
if ACTIVE_POOL not in STRATEGY_POOLS:
    raise ValueError(
        f"ACTIVE_POOL '{ACTIVE_POOL}' not found. "
        f"Available: {list(STRATEGY_POOLS.keys())}"
    )
_pool = STRATEGY_POOLS[ACTIVE_POOL]
EQUITY_TICKERS = list(_pool["RISK_ON"])
TICKER_BONDS = _pool["BOND"]
ALL_TICKERS = EQUITY_TICKERS + [TICKER_BONDS]


# ============================================================================
# FUNCTIONS
# ============================================================================

def get_price_data(ticker: str):
    """
    Download adjusted close prices from Yahoo Finance.

    Returns:
        pandas Series of adjusted close prices.

    Raises:
        ConnectionError: If no data returned.
        ValueError: If insufficient data or data is stale.
    """
    data = yf.download(
        ticker,
        period=DOWNLOAD_PERIOD,
        interval="1d",
        auto_adjust=False,
        progress=False,
        multi_level_index=False,
    )

    if data.empty:
        raise ConnectionError(
            f"No data from Yahoo Finance for {ticker}. "
            f"Check internet connection and ticker validity."
        )

    if "Adj Close" in data.columns:
        closes = data["Adj Close"].dropna()
    else:
        closes = data["Close"].dropna()

    if len(closes) < MIN_SESSIONS:
        raise ValueError(
            f"{ticker}: got {len(closes)} sessions, need {MIN_SESSIONS}."
        )

    last_date = closes.index[-1].date()
    days_stale = (datetime.date.today() - last_date).days
    if days_stale > 5:
        raise ValueError(
            f"{ticker}: data stale - last date {last_date} ({days_stale} days ago)."
        )

    return closes


def compute_momentum(closes) -> float:
    """Compute 12-month (252 trading day) momentum: close[-1] / close[-253] - 1."""
    price_now = float(closes.iloc[-1])
    price_past = float(closes.iloc[-(MOMENTUM_LOOKBACK + 1)])
    return price_now / price_past - 1.0


def compute_dual_momentum(closes) -> dict:
    """
    Compute 12M momentum for three reference points:
    - Today (last available date)
    - Month-start (first trading day of current month)
    - Month-end / Strict GEM (last trading day of previous month)

    Returns dict with today_mom, month_start_mom, month_end_mom
    and corresponding date pairs for manual verification.
    """
    # --- Today momentum ---
    price_now = float(closes.iloc[-1])
    today_date = closes.index[-1].date()
    today_lb_idx = max(len(closes) - 1 - MOMENTUM_LOOKBACK, 0)
    today_lb_date = closes.index[today_lb_idx].date()
    today_mom = price_now / float(closes.iloc[today_lb_idx]) - 1.0

    result = {
        "today_mom": today_mom,
        "today_price_date": today_date,
        "today_lookback_date": today_lb_date,
        "month_start_mom": None,
        "month_start_date": None,
        "month_start_lookback_date": None,
        "month_end_mom": None,
        "month_end_date": None,
        "month_end_lookback_date": None,
    }

    now = datetime.date.today()

    # --- Month-start momentum (first trading day of current month) ---
    mask_ms = (closes.index.year == now.year) & (closes.index.month == now.month)
    month_days = closes[mask_ms]
    if not month_days.empty:
        first_idx = closes.index.get_loc(month_days.index[0])
        price_ms = float(closes.iloc[first_idx])
        ms_lb_idx = max(first_idx - MOMENTUM_LOOKBACK, 0)
        result["month_start_mom"] = price_ms / float(closes.iloc[ms_lb_idx]) - 1.0
        result["month_start_date"] = closes.index[first_idx].date()
        result["month_start_lookback_date"] = closes.index[ms_lb_idx].date()

    # --- Month-end momentum (last trading day of previous month = Strict GEM) ---
    prev_year = now.year - 1 if now.month == 1 else now.year
    prev_month = 12 if now.month == 1 else now.month - 1
    mask_me = (closes.index.year == prev_year) & (closes.index.month == prev_month)
    prev_month_days = closes[mask_me]
    if not prev_month_days.empty:
        last_idx = closes.index.get_loc(prev_month_days.index[-1])
        price_me = float(closes.iloc[last_idx])
        me_lb_idx = max(last_idx - MOMENTUM_LOOKBACK, 0)
        result["month_end_mom"] = price_me / float(closes.iloc[me_lb_idx]) - 1.0
        result["month_end_date"] = closes.index[last_idx].date()
        result["month_end_lookback_date"] = closes.index[me_lb_idx].date()

    return result


def decide_asset(momentum: dict[str, float], previous_asset: str | None) -> str:
    """
    Apply GEM dual momentum rules.

    1. Relative: pick equity ETF with highest momentum (first in list wins ties).
    2. Churn filter: if previous equity is close enough, stay.
    3. Absolute: if best equity momentum <= 0, switch to BOND.
    """
    best_equity = max(EQUITY_TICKERS, key=lambda t: momentum[t])
    best_mom = momentum[best_equity]

    # Churn filter
    if (
        previous_asset in EQUITY_TICKERS
        and previous_asset != best_equity
        and momentum[previous_asset] > 0
    ):
        diff = abs(best_mom - momentum[previous_asset])
        if diff < CHURN_THRESHOLD:
            best_equity = previous_asset
            best_mom = momentum[best_equity]

    # Absolute momentum filter
    if best_mom <= 0:
        return TICKER_BONDS
    return best_equity


def get_last_decision() -> str | None:
    """Read the most recent decision for ACTIVE_POOL from CSV. Returns ticker or None."""
    if not os.path.isfile(DECISION_LOG):
        return None

    last = None
    try:
        with open(DECISION_LOG, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return None
            try:
                pool_idx = header.index("active_pool")
                asset_idx = header.index("selected_asset")
            except ValueError:
                return None
            for row in reader:
                if len(row) > max(pool_idx, asset_idx):
                    if row[pool_idx] == ACTIVE_POOL:
                        last = row[asset_idx]
    except OSError:
        pass
    return last


def save_decision(
    run_id: str,
    selected: str,
    momentum_today: dict[str, float],
    momentum_month_start: dict[str, float | None] | None = None,
    momentum_month_end: dict[str, float | None] | None = None,
) -> None:
    """Append decision row to decision_history_signal.csv."""
    file_exists = os.path.isfile(DECISION_LOG)
    with open(DECISION_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "date", "run_id", "active_pool",
                "selected_asset", "momentum_today",
                "momentum_month_start", "momentum_month_end",
            ])

        def _fmt_mom(mom_dict):
            if not mom_dict:
                return ""
            return "|".join(
                f"{t}:{mom_dict[t]:+.6f}" if mom_dict[t] is not None
                else f"{t}:N/A"
                for t in ALL_TICKERS
            )

        writer.writerow([
            datetime.date.today().isoformat(),
            run_id,
            ACTIVE_POOL,
            selected,
            _fmt_mom(momentum_today),
            _fmt_mom(momentum_month_start),
            _fmt_mom(momentum_month_end),
        ])


def write_heartbeat(run_id: str, selected: str) -> None:
    """Write heartbeat_signal.json with current run status."""
    heartbeat = {
        "last_run": datetime.datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "active_pool": ACTIVE_POOL,
        "selected_asset": selected,
    }
    with open(HEARTBEAT_FILE, "w", encoding="utf-8") as f:
        json.dump(heartbeat, f, indent=2)


def print_summary(
    momentum_today: dict[str, float],
    momentum_month_start: dict[str, float | None],
    momentum_month_end: dict[str, float | None],
    dual_details: dict[str, dict],
    previous: str | None,
    selected: str,
    decision_label: str,
) -> None:
    """Print formatted signal summary to console."""
    changed = (previous != selected) if previous else True

    # Build decision momentum for sorting and > 0? flag
    if STRICT_GEM_MODE:
        decision_mom = {
            t: momentum_month_end[t]
            if momentum_month_end.get(t) is not None
            else momentum_today[t]
            for t in ALL_TICKERS
        }
    else:
        decision_mom = momentum_today

    print(f"\n{'=' * 78}")
    print(f"  GEM SIGNAL - {ACTIVE_POOL}")
    if STRICT_GEM_MODE:
        print(f"  Mode: STRICT GEM (decisions based on month-end momentum)")
    print(f"{'=' * 78}")

    print(f"\n  ETFs in pool:")
    print(f"    RISK_ON: {', '.join(EQUITY_TICKERS)}")
    print(f"    BOND:    {TICKER_BONDS}")

    # Sort by decision momentum
    sorted_tickers = sorted(
        ALL_TICKERS, key=lambda t: decision_mom[t], reverse=True
    )

    star = "*" if STRICT_GEM_MODE else ""
    print(f"\n  {'Rank':<6}{'Ticker':<12}{'Today 12M':>12}{'M-Start':>12}{'Strict GEM' + star:>14}  {'> 0?':<6}")
    print(f"  {'-' * 66}")
    for rank, ticker in enumerate(sorted_tickers, 1):
        t_mom = momentum_today[ticker]
        ms_mom = momentum_month_start.get(ticker)
        me_mom = momentum_month_end.get(ticker)
        flag = "YES" if decision_mom[ticker] > 0 else "NO"
        label = " [BOND]" if ticker == TICKER_BONDS else ""
        t_str = f"{t_mom * 100:+.2f}%"
        ms_str = f"{ms_mom * 100:+.2f}%" if ms_mom is not None else "N/A"
        me_str = f"{me_mom * 100:+.2f}%" if me_mom is not None else "N/A"
        print(f"  {rank:<6}{ticker:<12}{t_str:>12}{ms_str:>12}{me_str:>14}  {flag:<6}{label}")

    print(f"\n  Decision based on: {decision_label}")

    # Decision
    print(f"\n  Previous: {previous or '(first run)'}")
    print(f"  Signal:   {selected}")
    print(f"  Change:   {'YES' if changed else 'NO'}")

    if changed and previous and previous != selected:
        if selected == TICKER_BONDS:
            print(f"  Action:   EXIT equities -> BONDS ({TICKER_BONDS})")
        elif previous == TICKER_BONDS:
            print(f"  Action:   EXIT bonds -> EQUITIES ({selected})")
        else:
            print(f"  Action:   ROTATE {previous} -> {selected}")
    elif not changed:
        print(f"  Action:   HOLD {selected}")
    else:
        print(f"  Action:   INITIAL BUY {selected}")

    # Calculation dates for manual verification
    print(f"\n  Calculation dates:")
    for ticker in ALL_TICKERS:
        d = dual_details[ticker]
        t_line = f"{d['today_price_date']} vs {d['today_lookback_date']}"
        if d["month_start_date"]:
            ms_line = f"{d['month_start_date']} vs {d['month_start_lookback_date']}"
        else:
            ms_line = "N/A"
        if d["month_end_date"]:
            me_line = f"{d['month_end_date']} vs {d['month_end_lookback_date']}"
        else:
            me_line = "N/A"
        print(f"    {ticker:<12} Today:      {t_line}")
        print(f"    {'':12} M-Start:    {ms_line}")
        print(f"    {'':12} Strict GEM: {me_line}")

    print(f"\n{'=' * 78}")


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    """Generate GEM momentum signal."""
    run_id = uuid.uuid4().hex[:8]

    print(f"\n=== GEM Signal Generator [RUN {run_id}] ===")
    print(f"Pool: {ACTIVE_POOL}  |  Date: {datetime.date.today()}")

    # Step 1: Download prices and compute momentum (today + month-start + month-end)
    momentum_today = {}
    momentum_month_start = {}
    momentum_month_end = {}
    dual_details = {}
    print(f"\nFetching price data...")
    for ticker in ALL_TICKERS:
        print(f"  {ticker}...", end=" ", flush=True)
        closes = get_price_data(ticker)
        dual = compute_dual_momentum(closes)
        momentum_today[ticker] = dual["today_mom"]
        momentum_month_start[ticker] = dual["month_start_mom"]
        momentum_month_end[ticker] = dual["month_end_mom"]
        dual_details[ticker] = dual
        sessions = len(closes)
        me_str = (
            f"{dual['month_end_mom']:+.4f}"
            if dual["month_end_mom"] is not None
            else "N/A"
        )
        print(f"ok ({sessions} sessions, today: {dual['today_mom']:+.4f}, strict: {me_str})")

    # Step 2: Get previous decision (for churn filter)
    previous = get_last_decision()

    # Step 3: Build decision momentum based on mode
    if STRICT_GEM_MODE:
        decision_momentum = {}
        for t in ALL_TICKERS:
            me = momentum_month_end[t]
            if me is not None:
                decision_momentum[t] = me
            else:
                decision_momentum[t] = momentum_today[t]
                print(f"  WARNING: {t} month-end momentum N/A, falling back to today")
        decision_label = "Strict GEM (Month-End)"
    else:
        decision_momentum = momentum_today
        decision_label = "Today 12M"

    selected = decide_asset(decision_momentum, previous)

    # Step 4: Print summary
    print_summary(
        momentum_today, momentum_month_start, momentum_month_end,
        dual_details, previous, selected, decision_label,
    )

    # Step 5: Save results
    save_decision(
        run_id, selected, momentum_today, momentum_month_start, momentum_month_end,
    )
    write_heartbeat(run_id, selected)
    print(f"  Saved: {DECISION_LOG}")
    print(f"  Saved: {HEARTBEAT_FILE}")
    print()


if __name__ == "__main__":
    try:
        main()
    except ConnectionError as e:
        print(f"\nCONNECTION ERROR: {e}")
    except ValueError as e:
        print(f"\nDATA ERROR: {e}")
    except KeyboardInterrupt:
        print("\n\nInterrupted.")
    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {e}")
