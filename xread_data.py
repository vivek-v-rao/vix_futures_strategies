"""
Read the VIX futures and index csv files of this project into pandas frames.

read_monthly_contracts reads the per contract files in vix_futures_contracts, read_vix_indexes the CBOE
index histories in vix_spot, and add_returns works out each contract's daily returns and how many
trading days it has left. Nothing here downloads anything.
"""
import numpy as np
import pandas as pd
from pathlib import Path

def read_monthly_contracts(data_dir="vix_futures_contracts") -> pd.DataFrame:
    """Return all monthly contract records, one row per (Trade Date, Expiry)."""
    files = sorted(Path(data_dir).glob("VX_*_monthly.csv"))
    if not files:
        raise FileNotFoundError(f"no VX_*_monthly.csv files in {data_dir}")
    df = pd.concat(
        (pd.read_csv(f, parse_dates=["Trade Date", "Expiry"]) for f in files),
        ignore_index=True,
    )
    return df.sort_values(["Trade Date", "Expiry"], ignore_index=True)

def read_vix_indexes(path="vix_spot/vix_indexes.csv", symbols=("VIX",)) -> pd.DataFrame:
    """Return daily CBOE index values (written by `vixutil -c vix_spot/vix_indexes.csv`) with one
    column per symbol of closing values, indexed by trade date. symbols=None returns all symbols."""
    df = pd.read_csv(path, index_col=0, parse_dates=["Trade Date"])
    if symbols is not None:
        df = df[df["Symbol"].isin(symbols)]
    return df.pivot(index="Trade Date", columns="Symbol", values="Close")

def expired_rows(df: pd.DataFrame) -> pd.Series:
    """True for the rows of contracts that have expired: the Expired flag of the contract's last row is
    set, or its expiry is on or before the last trade date in df (a few early contracts have no data
    for their final days, so their flag is not set)."""
    flag = df.groupby("Expiry")["Expired"].transform("last").astype(bool)
    return flag | (df["Expiry"] <= df["Trade Date"].max())

MONTH_CODES = "FGHJKMNQUVXZ"

def contract_symbol(expiry: pd.Timestamp) -> str:
    """CBOE symbol of the VIX future expiring in that month, such as VXU26 for September 2026."""
    return f"VX{MONTH_CODES[expiry.month - 1]}{expiry.year % 100:02d}"

def add_returns(df: pd.DataFrame, log_returns: bool = False) -> pd.DataFrame:
    """Add daily close-to-close returns and trading days until each contract's last close.

    The trading calendar is the set of all trade dates in df. A return is only computed when the
    contract's previous close was on the previous trading day, so gaps in a contract's history
    do not produce multi-day returns. Days_To_Last_Close is 0 on the day of the last close.
    Day_Number numbers the trading days of the calendar 0, 1, 2, ...
    Only contracts that have expired are used, so the last close is final.
    """
    df = df.sort_values(["Expiry", "Trade Date"])
    df = df[expired_rows(df)].copy()

    calendar = pd.Series(sorted(df["Trade Date"].unique()))
    day_number = pd.Series(calendar.index, index=calendar.values)
    df["Day_Number"] = df["Trade Date"].map(day_number)

    g = df.groupby("Expiry")
    df["Days_To_Last_Close"] = g["Day_Number"].transform("max") - df["Day_Number"]
    prev_close = g["Close"].shift(1)
    consecutive = (df["Day_Number"] - g["Day_Number"].shift(1)) == 1
    ratio = df["Close"] / prev_close
    ret = np.log(ratio) if log_returns else ratio - 1
    df["Return"] = ret.where(consecutive)
    return df
def add_carry(returns: pd.DataFrame, vix_close: pd.Series, log_returns: bool = False) -> pd.DataFrame:
    """Add the carry per day of each contract relative to the next contract closer to expiration.

    returns is the output of add_returns. On each trade date the contracts are ordered by expiry;
    the nearer "contract" of the front contract is spot VIX (vix_close, indexed by trade date) with
    0 days to its last close. For contract k with close F_k and T_k trading days to its last close,
    and nearer contract j,
        Carry = -(F_k - F_j) / ((T_k - T_j) * F_k)     (simple returns)
        Carry = -log(F_k / F_j) / (T_k - T_j)          (log returns)
    which is the daily return from rolling down the curve if its shape did not change.
    Front is True when the nearer price is spot VIX. Lag_Carry, Lag_Front and Lag_Days_To_Last_Close
    are the values on the previous trading day, so they are known before the day's Return; they are
    NaN where Return is NaN.
    """
    d = returns.sort_values(["Trade Date", "Expiry"]).copy()
    by_date = d.groupby("Trade Date")
    near_close = by_date["Close"].shift(1)
    near_days = by_date["Days_To_Last_Close"].shift(1)
    front = near_close.isna()
    near_close = near_close.where(~front, d["Trade Date"].map(vix_close))
    near_days = near_days.where(~front, 0)

    gap = (d["Days_To_Last_Close"] - near_days).where(lambda x: x > 0)
    if log_returns:
        carry = -np.log(d["Close"] / near_close) / gap
    else:
        carry = -(d["Close"] - near_close) / (gap * d["Close"])
    d["Front"] = front
    d["Carry"] = carry

    d = d.sort_values(["Expiry", "Trade Date"])
    by_contract = d.groupby("Expiry")
    has_return = d["Return"].notna()
    d["Lag_Carry"] = by_contract["Carry"].shift(1).where(has_return)
    d["Lag_Front"] = by_contract["Front"].shift(1).where(has_return)
    d["Lag_Days_To_Last_Close"] = by_contract["Days_To_Last_Close"].shift(1).where(has_return)
    return d

def tenor_grid(df: pd.DataFrame) -> pd.DataFrame:
    """The tenor each contract stood at on each trade date, one column per contract.

    Tenor 1 is the contract nearest to expiration that can still be held, so a contract at its last
    close is not counted and the one behind it moves up that day. This is the numbering the backtests
    use, and it is what says which contract a weight in a given slot was actually in.
    """
    c = add_returns(df)
    c = c[c["Days_To_Last_Close"] > 0].sort_values(["Trade Date", "Expiry"])
    c["Tenor"] = c.groupby("Trade Date").cumcount() + 1
    return c.pivot(index="Trade Date", columns="Expiry", values="Tenor")

def carry_by_contract(df: pd.DataFrame, vix_close: pd.Series) -> pd.DataFrame:
    """The carry per day of every contract on every trade date, one column per contract."""
    carry = add_carry(add_returns(df), vix_close)
    return carry.pivot(index="Trade Date", columns="Expiry", values="Carry")

def split_return(positions: pd.DataFrame, carry: pd.DataFrame, returns: pd.Series) -> pd.DataFrame:
    """Split each day's return into the carry of the position held and what the curve did.

    The position held from the previous close earns today's return, so its carry is that position and
    those carries, both as of the previous close: what the position would have earned had the curve
    kept its shape. The rest is the curve moving, and the two add back to the return exactly.

    Carry is the part that can be seen in advance. For a position held long, it is the price of the
    insurance; for one held short, it is the premium collected.
    """
    held = positions.shift(1)
    aligned = carry.reindex(index=held.index, columns=held.columns).shift(1)
    day_carry = (held * aligned).sum(axis=1).reindex(returns.index).fillna(0.0)
    return pd.DataFrame({"Carry": day_carry, "Curve": returns - day_carry, "Return": returns})

def expand_positions(daily: pd.DataFrame) -> pd.DataFrame:
    """The Positions column of a backtest as a frame of one column per contract (and SPY for a hedge).

    Each row is the book held after that day's trades, as a share of equity, so the next day's return
    is the previous row times each contract's return. Contracts not held are 0. A portfolio can add the
    books of several strategies and charge trading costs on the change in the total, which nets the
    trades they make against each other in the same contract.
    """
    books = daily["Positions"]
    contracts = sorted({name for book in books for name in book}, key=str)
    frame = pd.DataFrame(0.0, index=daily.index, columns=contracts)
    for date, book in books.items():
        for name, weight in book.items():
            frame.at[date, name] = weight
    return frame

def carry_summary(parts: pd.DataFrame, vix: pd.Series, trading_days_per_year: float = 252) -> pd.DataFrame:
    """What the carry of the position has been, and what it is now.

    Carry is quoted annualized. It scales with the level of VIX, so the same points of carry mean
    something different at VIX 12 and VIX 40; the second column divides by VIX, and the percentile is
    taken on that, since it is what says whether the position is dear by its own standards.
    """
    carry = parts["Carry"] * trading_days_per_year
    relative = (parts["Carry"] / vix.reindex(parts.index)).dropna() * trading_days_per_year
    latest = carry.index[-1]
    rows = {
        "Average": {"Carry_Ann": carry.mean(), "Carry_Over_VIX": relative.mean()},
        "Median": {"Carry_Ann": carry.median(), "Carry_Over_VIX": relative.median()},
        "Most negative": {"Carry_Ann": carry.min(), "Carry_Over_VIX": relative.min()},
        "Least negative": {"Carry_Ann": carry.max(), "Carry_Over_VIX": relative.max()},
        f"Now ({latest.date()})": {"Carry_Ann": carry.iloc[-1], "Carry_Over_VIX": relative.iloc[-1]},
    }
    frame = pd.DataFrame(rows).T
    frame["Percentile"] = np.nan
    frame.loc[f"Now ({latest.date()})", "Percentile"] = (relative <= relative.iloc[-1]).mean() * 100
    return frame

def carry_by_year(parts: pd.DataFrame) -> pd.DataFrame:
    """Each year's return split into the carry of the position and what the curve did.

    The pieces are added up day by day, so they do not compound into the year's return exactly; they
    show where the year came from, not an exact attribution.
    """
    yearly = parts.groupby(parts.index.year).sum()
    yearly["Compounded"] = (1 + parts["Return"]).groupby(parts.index.year).prod() - 1
    yearly.index.name = "Year"
    return yearly

def price_series(df: pd.DataFrame, column: str = "Settle", limit: float = 0.5) -> pd.Series:
    """The price column to work from, with the close standing in where it cannot be believed.

    CBOE recorded no settlement price at all for the contracts of 2013, and on 2008-04-21 recorded 1.00
    for a contract that closed at 22.67, which would read as a 96% fall and a 2000% rise the next day.
    So the chosen column is used where it is positive and within limit of the close, and the close is
    used otherwise. One row in the whole history is caught by the second test; the next largest gap
    between the two is 20%, which is ordinary for a contract that hardly trades.
    """
    if column not in df:
        return df["Close"]
    chosen = df[column]
    believable = (chosen > 0) & ((chosen / df["Close"] - 1).abs() <= limit)
    return chosen.where(believable, df["Close"])
