"""
GEM Signal Dashboard - Streamlit Web UI
Run: streamlit run app.py
"""

import json
import os
import uuid

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import yfinance as yf

import gem_signal
from gem_signal import (
    STRATEGY_POOLS,
    MOMENTUM_LOOKBACK,
    STRICT_GEM_MODE,
    get_price_data,
    compute_dual_momentum,
    decide_asset,
    get_last_decision,
    save_decision,
    write_heartbeat,
)

st.set_page_config(page_title="GEM Signal Dashboard", layout="wide")

SETTINGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")


# ============================================================================
# SETTINGS
# ============================================================================

def load_settings() -> dict:
    """Load settings from JSON file; create with defaults if missing."""
    defaults = {"active_pool": list(STRATEGY_POOLS.keys())[0]}
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Validate that saved pool still exists
        if data.get("active_pool") not in STRATEGY_POOLS:
            data["active_pool"] = defaults["active_pool"]
        return data
    except (FileNotFoundError, json.JSONDecodeError):
        save_settings(defaults)
        return defaults


def save_settings(settings: dict) -> None:
    """Write settings to JSON file."""
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)


# ============================================================================
# HELPERS
# ============================================================================

def _apply_pool(name: str) -> tuple[list[str], str, list[str]]:
    """Set gem_signal module globals for the selected pool."""
    pool = STRATEGY_POOLS[name]
    gem_signal.ACTIVE_POOL = name
    gem_signal.EQUITY_TICKERS = list(pool["RISK_ON"])
    gem_signal.TICKER_BONDS = pool["BOND"]
    gem_signal.ALL_TICKERS = gem_signal.EQUITY_TICKERS + [gem_signal.TICKER_BONDS]
    return gem_signal.EQUITY_TICKERS, gem_signal.TICKER_BONDS, gem_signal.ALL_TICKERS


def _load_history(pool_name: str, n: int = 6) -> pd.DataFrame | None:
    """Load last N decision rows for pool from CSV (most recent first)."""
    path = gem_signal.DECISION_LOG
    if not os.path.isfile(path):
        return None
    try:
        df = pd.read_csv(path)
        if "active_pool" in df.columns:
            df = df[df["active_pool"] == pool_name]
        if df.empty:
            return None
        return df.tail(n).iloc[::-1].reset_index(drop=True)
    except Exception:
        return None


def _color_momentum(val):
    """Styler: green for positive momentum, red for negative."""
    if isinstance(val, str):
        if val.startswith("+"):
            return "color: #28a745; font-weight: bold"
        if val.startswith("-"):
            return "color: #dc3545; font-weight: bold"
    return ""


def _color_difference(val):
    """Styler: orange warning for difference > 1%."""
    if isinstance(val, str) and val not in ("N/A", ""):
        try:
            num = float(val.replace("%", "").replace("+", ""))
            if abs(num) > 1.0:
                return "color: #fd7e14; font-weight: bold"
        except ValueError:
            pass
    return ""


def compute_monthly_snapshots(
    all_tickers: list[str], lookback: int = 252,
) -> pd.DataFrame | None:
    """Compute 12M momentum at first trading day of each of last 12 months."""
    frames = {}
    for ticker in all_tickers:
        data = yf.download(
            ticker, period="800d", interval="1d",
            auto_adjust=False, progress=False, multi_level_index=False,
        )
        if data.empty:
            continue
        col = "Adj Close" if "Adj Close" in data.columns else "Close"
        frames[ticker] = data[col].dropna()

    if not frames:
        return None

    ref = next(iter(frames.values()))
    first_days = []
    for (_y, _m), grp in ref.groupby([ref.index.year, ref.index.month]):
        first_days.append(grp.index[0])
    first_days.sort()
    checkpoints = first_days[-12:]

    rows = []
    for cp in checkpoints:
        row = {"Month": cp.strftime("%Y-%m")}
        for ticker, series in frames.items():
            if cp not in series.index:
                row[ticker] = None
                continue
            pos = series.index.get_loc(cp)
            if pos < lookback:
                row[ticker] = None
                continue
            row[ticker] = float(series.iloc[pos]) / float(series.iloc[pos - lookback]) - 1.0
        rows.append(row)

    return pd.DataFrame(rows) if rows else None


# ============================================================================
# PAGE CONFIG
# ============================================================================

st.title("GEM Signal Dashboard")

# --- Controls row ---
settings = load_settings()
pool_names = list(STRATEGY_POOLS.keys())
saved_index = pool_names.index(settings["active_pool"])

ctl_pool, ctl_btn = st.columns([2, 1])
with ctl_pool:
    pool_name = st.selectbox("Strategy Pool", pool_names, index=saved_index)

# Persist pool change
if pool_name != settings["active_pool"]:
    settings["active_pool"] = pool_name
    save_settings(settings)

equity_tickers, bond_ticker, all_tickers = _apply_pool(pool_name)
with ctl_btn:
    st.write("")  # vertical spacer to align with selectbox
    compute = st.button("Oblicz sygnal", type="primary", use_container_width=True)

st.caption(f"RISK_ON: {', '.join(equity_tickers)}  |  BOND: {bond_ticker}")

# --- Compute signal ---
if compute:
    mom_today = {}
    mom_ms = {}
    mom_me = {}
    prices = {}
    with st.spinner("Pobieranie danych z Yahoo Finance..."):
        for ticker in all_tickers:
            try:
                closes = get_price_data(ticker)
                dual = compute_dual_momentum(closes)
                mom_today[ticker] = dual["today_mom"]
                mom_ms[ticker] = dual["month_start_mom"]
                mom_me[ticker] = dual["month_end_mom"]
                prices[ticker] = closes
            except (ConnectionError, ValueError) as e:
                st.error(f"Blad pobierania {ticker}: {e}")
                st.stop()

    # Build decision momentum based on mode
    if STRICT_GEM_MODE:
        decision_mom = {
            t: mom_me[t] if mom_me.get(t) is not None else mom_today[t]
            for t in all_tickers
        }
    else:
        decision_mom = mom_today

    previous = get_last_decision()
    selected = decide_asset(decision_mom, previous)

    with st.spinner("Obliczanie snapshotów miesięcznych..."):
        snapshots = compute_monthly_snapshots(all_tickers)

    st.session_state.update(
        momentum_today=mom_today,
        momentum_month_start=mom_ms,
        momentum_month_end=mom_me,
        decision_momentum=decision_mom,
        prices=prices,
        selected=selected,
        previous=previous,
        snapshots=snapshots,
        run_id=uuid.uuid4().hex[:8],
        pool=pool_name,
        saved=False,
    )

# --- Guard: nothing computed yet ---
if "momentum_today" not in st.session_state or st.session_state.get("pool") != pool_name:
    st.divider()
    st.info("Wybierz pool i kliknij **Oblicz sygnal** aby wygenerowac sygnal GEM.")
    st.stop()

# --- Unpack session state ---
mom_today = st.session_state.momentum_today
mom_ms = st.session_state.momentum_month_start
mom_me = st.session_state.momentum_month_end
decision_mom = st.session_state.decision_momentum
prices = st.session_state.prices
selected = st.session_state.selected
previous = st.session_state.previous
best_eq = max(equity_tickers, key=lambda t: decision_mom[t])
changed = (previous != selected) if previous else True


# ============================================================================
# TOP ROW: 4 metric columns
# ============================================================================

st.divider()
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric("Final Decision", selected)
with k2:
    label = "Best Strict GEM" if STRICT_GEM_MODE else "Best Momentum"
    st.metric(
        label,
        f"{decision_mom[best_eq] * 100:+.2f}%",
        delta=best_eq,
        delta_color="off",
    )
with k3:
    st.metric("Active Pool", pool_name)
with k4:
    prev_label = f"poprz. {previous}" if previous else "pierwsza decyzja"
    st.metric(
        "Zmiana vs poprzednia",
        "TAK" if changed else "NIE",
        delta=prev_label,
        delta_color="off",
    )

st.divider()


# ============================================================================
# MIDDLE: Ranking (left) + Chart (right)
# ============================================================================

col_rank, col_chart = st.columns([2, 3])

# --- LEFT: Ranking ETF ---
with col_rank:
    st.subheader("Ranking ETF")

    me_col = "Strict GEM *" if STRICT_GEM_MODE else "Month-End 12M"
    # Sort by decision momentum
    sorted_tickers = sorted(all_tickers, key=lambda t: decision_mom[t], reverse=True)
    rows = []
    for rank, ticker in enumerate(sorted_tickers, 1):
        t_mom = mom_today[ticker]
        ms_mom = mom_ms.get(ticker)
        me_mom = mom_me.get(ticker)
        ms_str = f"{ms_mom * 100:+.2f}%" if ms_mom is not None else "N/A"
        me_str = f"{me_mom * 100:+.2f}%" if me_mom is not None else "N/A"
        # Drift = Today vs Strict GEM (how much changed since decision point)
        if me_mom is not None:
            drift = t_mom - me_mom
            drift_str = f"{drift * 100:+.2f}%"
        else:
            drift_str = "N/A"
        rows.append({
            "Rank": rank,
            "Ticker": ticker,
            "Today 12M": f"{t_mom * 100:+.2f}%",
            "M-Start 12M": ms_str,
            me_col: me_str,
            "Drift": drift_str,
        })
    df_rank = pd.DataFrame(rows)
    mom_cols = ["Today 12M", "M-Start 12M", me_col]
    styled = df_rank.style.map(
        _color_momentum, subset=mom_cols
    ).map(_color_difference, subset=["Drift"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    if STRICT_GEM_MODE:
        st.caption("\\* Decision based on **Strict GEM** (month-end momentum)")
    else:
        st.caption("Decision based on **Today 12M** momentum")

    # Decision highlight box
    if selected == bond_ticker:
        st.warning(f"**{selected}** — Defensive (equity momentum <= 0)")
    else:
        st.success(f"**{selected}** — Best Risk-On ETF")

# --- RIGHT: 12M Price Chart ---
with col_chart:
    st.subheader("Wykres 12M")
    chart_ticker = st.selectbox("ETF", all_tickers, key="chart_etf")

    if chart_ticker in prices:
        series = prices[chart_ticker].iloc[-MOMENTUM_LOOKBACK:]
        line_color = "#28a745" if decision_mom[chart_ticker] > 0 else "#dc3545"

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(series.index, series.values, linewidth=1.5, color=line_color)
        ax.fill_between(
            series.index, series.values, alpha=0.07, color=line_color
        )
        ax.set_title(f"12M Price Chart - {chart_ticker}", fontsize=13)
        ax.set_ylabel("Adj Close")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        fig.autofmt_xdate()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning("Brak danych cenowych dla wybranego ETF.")

st.divider()


# ============================================================================
# ANNUAL SNAPSHOT PERFORMANCE
# ============================================================================

df_snap = st.session_state.get("snapshots")
if df_snap is not None and not df_snap.empty:
    st.subheader("Annual Snapshot Performance (Monthly Checkpoints)")
    ticker_cols = [c for c in df_snap.columns if c != "Month"]

    # --- Pivot table with colored momentum + star for best ---
    display_snap = df_snap[["Month"] + ticker_cols].copy()
    for idx in display_snap.index:
        vals = {
            c: display_snap.at[idx, c]
            for c in ticker_cols
            if pd.notna(display_snap.at[idx, c])
        }
        if vals:
            best_col = max(vals, key=vals.get)
            for c in ticker_cols:
                v = display_snap.at[idx, c]
                if pd.notna(v):
                    fmt = f"{v * 100:+.2f}%"
                    if c == best_col:
                        fmt += " *"
                    display_snap.at[idx, c] = fmt
                else:
                    display_snap.at[idx, c] = "N/A"

    styled_snap = display_snap.style.map(_color_momentum, subset=ticker_cols)
    st.dataframe(styled_snap, use_container_width=True, hide_index=True)
    st.caption("* = best ETF in given month")

    # --- Line chart: rolling 12M momentum per ETF ---
    fig_snap, ax_snap = plt.subplots(figsize=(10, 4))
    for ticker in ticker_cols:
        values = pd.to_numeric(df_snap[ticker], errors="coerce")
        mask = values.notna()
        ax_snap.plot(
            df_snap.loc[mask, "Month"],
            values[mask] * 100,
            marker="o", markersize=4, linewidth=1.5, label=ticker,
        )
    ax_snap.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax_snap.set_ylabel("12M Momentum (%)")
    ax_snap.set_title("Rolling 12M Momentum at Monthly Checkpoints")
    ax_snap.legend()
    ax_snap.grid(True, alpha=0.3)
    fig_snap.autofmt_xdate()
    fig_snap.tight_layout()
    st.pyplot(fig_snap)
    plt.close(fig_snap)

    st.divider()


# ============================================================================
# BOTTOM: History + Save
# ============================================================================

col_hist, col_save = st.columns([3, 1])

with col_hist:
    st.subheader("Historia decyzji (ostatnie 6)")
    df_hist = _load_history(pool_name, n=6)
    if df_hist is not None:
        display = df_hist[["date", "run_id", "selected_asset"]].copy()
        display.columns = ["Data", "Run ID", "Decyzja"]
        st.dataframe(display, use_container_width=True, hide_index=True)
    else:
        st.caption("Brak zapisanych decyzji dla tego poolu.")

with col_save:
    st.subheader("Zapis")
    if st.session_state.get("saved"):
        st.success("Decyzja zapisana do CSV i heartbeat.")
    elif st.button("Zapisz decyzje", use_container_width=True):
        rid = st.session_state.run_id
        save_decision(rid, selected, mom_today, mom_ms, mom_me)
        write_heartbeat(rid, selected)
        st.session_state.saved = True
        st.rerun()
