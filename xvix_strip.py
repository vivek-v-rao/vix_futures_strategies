"""
Hold a strip of VIX futures, long or short, rolled along the curve, and see what that did.

A strip is a run of consecutive contracts held together, first to last, with a fraction of the nearest
rolled into the furthest every trading day so the average maturity stays put. That is how the S&P VIX
futures indexes are built, and the default here is the mid term one that VIXM tracks, the contracts 4
to 7 months out at about five months' maturity. Held long it pays the volatility risk premium instead
of earning it, which is why it loses money on its own; what it offers is a negative beta to equities
that grows in a crash. --scale takes a negative multiple to sell the strip instead, which is the
opposite trade in every respect, and the commentary follows the sign.

The program prints, in order: what it holds, how closely it matches the fund holding the same strip
where one does, what it did on its own, what it did against SPY, and what a slice of it did for a
portfolio of SPY, sized in a fixed proportion and sized with the level of VIX.

It reads the csv files in this repository and downloads nothing.

Either tenor may be a range, so --tenors 1:3 1:3 runs the six positions from 1-1 to 3-3 one after
another and --single-contract 1:6 runs each of the six contracts held alone. The two add together and
a position named twice runs once, since a pair of equal tenors is the single contract. --widths keeps
only the strips of a given number of contracts, which is what keeps a wide sweep short enough to read.

usage: python xvix_strip.py [--tenors FIRST LAST] [--single-contract TENOR] [--widths N] [--scale X]
       [--cost C]
       [--slices W ...] [--overlay-size W]
       [--plot out.png] [--output daily.csv] [--data-dir path] [--vix-file path] [--spy-file path]
       [--rf-file path] [--vixm-file path] [--vxx-file path]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from xread_data import (carry_by_contract, contract_symbol, expand_positions, price_series,
                        read_monthly_contracts, read_vix_indexes, split_return, tenor_grid)
from xstrategy_index import backtest_index, backtest_single_contract

# the strips a fund holds, so the construction can be checked against something real: the name, the
# option carrying its prices, and the fee, which is what the difference from the index should come to
FUNDS = {(1, 2): ("VXX", "vxx_file", 0.0089), (4, 7): ("VIXM", "vixm_file", 0.0085)}

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
TRADING_DAYS = 252

# the rulers the stand-alone table measures with: the pandas period, and how many fall in a year
PERIODS = {"Daily": (None, TRADING_DAYS), "Weekly": ("W-FRI", 52), "Monthly": ("ME", 12),
           "Quarterly": ("QE", 4), "Annual": ("YE", 1)}

def ordinal(n: int) -> str:
    """1st, 2nd, 3rd, 4th, and the teens that break the pattern."""
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"

def int_range(text: str) -> list:
    """One number, "4", or an inclusive range of them, "1:3", as the list it names."""
    first, _, last = text.partition(":")
    try:
        start, stop = int(first), int(last or first)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number or a range of them like 1:3")
    if not 1 <= start <= stop:
        raise argparse.ArgumentTypeError(f"{text!r} must run upwards from 1")
    return list(range(start, stop + 1))

def sweep_pairs(tenors, singles, widths=None) -> list:
    """The positions to run, as (first, last) pairs of tenors with first no later than last.

    A pair of equal tenors is the single contract rolled once a month, one tenor leaving nothing to
    roll into, so the two ways of asking overlap: --tenors 1:3 1:3 already contains the contracts that
    --single-contract 1:3 names. They go into one set, so each position runs once however it was asked
    for, and the set is sorted so the runs read along the curve whatever order they were given in.

    widths keeps only the strips holding that many contracts, which is what makes a wide sweep worth
    printing: the pairs grow with the square of the tenors, and the widest of them overlap so heavily
    that they mostly restate each other, 1-6 and 1-7 sharing six contracts of seven. It prunes the
    pairs the ranges generate and leaves the contracts named by singles alone, since those were asked
    for one by one and every one of them is a strip of width one.
    """
    pairs = {(first, last) for first in tenors[0] for last in tenors[1] if first <= last}
    if widths:
        pairs = {(first, last) for first, last in pairs if last - first + 1 in widths}
    return sorted(pairs | {(tenor, tenor) for tenor in singles})

def suffixed(path, first: int, last: int) -> str:
    """A file name with the tenors put before the extension, so a sweep does not overwrite itself."""
    path = Path(path)
    return str(path.with_name(f"{path.stem}_{first}_{last}{path.suffix}"))

def position_detail(daily: pd.DataFrame, prices: pd.DataFrame, carry: pd.DataFrame,
                    tenors: pd.DataFrame, last_tenor: int) -> pd.DataFrame:
    """The contracts behind each day's weights: which ones they were, at what price, and what they did.

    Weight_k is the weight held in the contract that stood at tenor k at the previous close, because
    that is the book which earns the day's return. These columns follow that same book, so Weight_k
    times Return_k, added across the tenors, is the day's gross return.

    A row therefore reads: at yesterday's close you held Weight_k of contract Expiry_k, whose carry per
    day was Carry_k; today it returned Return_k and closed at Price_k. The weight and the carry are as
    of the previous close, the price and the return as of this one, which is the only arrangement in
    which the columns multiply out to the day's return and its carry.

    The exception is Weight_Other, which the backtest fills when a contract it holds had no quote on
    the previous day, so there is no tenor to put it under. Such contracts get no slot here either, for
    the same reason, and on those days the products fall short of the gross return by what they earned.
    It is a feature of the early years, when the far end of the curve did not trade every day.
    """
    book = expand_positions(daily).shift(1)
    held_tenors = tenors.reindex(index=book.index, columns=book.columns).shift(1)
    detail = {}
    for tenor in range(1, last_tenor + 1):
        at_tenor = held_tenors.eq(tenor) & book.ne(0) & book.notna()
        which = at_tenor.idxmax(axis=1).where(at_tenor.any(axis=1))
        rows = which.dropna()
        columns = {"Expiry": rows, "Symbol": rows.map(contract_symbol)}
        for name, grid in (("Price", prices), ("Carry", carry.shift(1)), ("Return", None)):
            frame = prices.pct_change(fill_method=None) if grid is None else grid
            lined = frame.reindex(index=rows.index, columns=prices.columns)
            columns[name] = pd.Series([lined.at[date, expiry] for date, expiry in rows.items()],
                                      index=rows.index)
        for name, column in columns.items():
            detail[f"{name}_{tenor}"] = column.reindex(book.index)
    return pd.DataFrame(detail, index=daily.index)

def extreme_days(daily: pd.DataFrame, parts: pd.DataFrame, frame: pd.DataFrame, detail: pd.DataFrame,
                 first: int, last: int, count: int, worst: bool = True) -> pd.DataFrame:
    """The days the position moved most, with what it was holding and what each contract did.

    Strip is the day's net return, split into the Carry it was always going to earn and the Curve
    moving, which is what makes a day extreme. SPY and VIX put the day in its market, and Pk is the
    closing price of the contract at tenor k, so the shape of the curve that day can be read against
    the spot beside it. Wk is the weight held there and Rk that contract's own return, so the products
    add to the day before costs.

    The contracts of a strip correlate 0.95 and above day to day, but on these days they do not move
    together: the near end moves furthest, and the weights say how much of the position was sitting
    there, which is set by where the roll had got to rather than by anything about the day.

    These rows are also the first place a bad price shows itself. A date that answers to nothing that
    happened in the market is worth looking up in the contract file before it is believed.
    """
    returns = daily["Net_Return"].reindex(frame.index)
    picked = (returns.nsmallest(count) if worst else returns.nlargest(count)).index
    table = pd.DataFrame({"Strip": returns.reindex(picked), "Carry": parts["Carry"].reindex(picked),
                          "Curve": parts["Curve"].reindex(picked), "SPY": frame["spy"].reindex(picked),
                          "VIX": frame["vix"].reindex(picked)})
    for tenor in range(first, last + 1):
        table[f"P{tenor}"] = detail[f"Price_{tenor}"].reindex(picked)
    for tenor in range(first, last + 1):
        table[f"W{tenor}"] = daily[f"Weight_{tenor}"].reindex(picked)
        table[f"R{tenor}"] = detail[f"Return_{tenor}"].reindex(picked)
    table.index.name = "Trade Date"
    return table

def closing_note(frame: pd.DataFrame, size: float) -> str:
    """What this run found, in a sentence, rather than what was true of one position years ago.

    Which years carry a slice does not depend on how large the slice is, only the amounts do, so the
    size used is the one the sized overlays aim at and the years would be the same at any other.

    The position's own figure is the compounded return, which for a series this volatile sits well
    below its mean: a short strip with a clearly positive mean can still have lost money over the
    sample, and the compounded figure is the one a holder is left with.
    """
    added = pd.Series({year: (1 + part["spy"] + size * part["strategy"]).prod()
                             - (1 + part["spy"]).prod()
                       for year, part in frame.groupby(frame.index.year)})
    best = added.nlargest(2)
    rest, years = added.drop(best.index).sum(), best.index.tolist()
    grew = (1 + frame["strategy"]).prod() ** (TRADING_DAYS / len(frame)) - 1
    return (f"the position {'made' if grew > 0 else 'lost'} {abs(grew):.1%} a year on its own, "
            f"compounding what averaged {frame['strategy'].mean() * TRADING_DAYS:+.1%}; a slice of "
            f"{size:.0%} on top of SPY helped in {(added > 0).sum()} of {len(added)} years, and rests "
            f"on {years[0]} and {years[1]} ({best.iloc[0]:+.1%} and {best.iloc[1]:+.1%}), without which "
            f"it {'added' if rest > 0 else 'cost'} {abs(rest):.1%} over the other {len(added) - 2} years")

def unwritable(path) -> str:
    """Why a file cannot be written, or an empty string if it can be.

    The work comes before the writing, so without this a run spends its minutes and then throws them
    away on the last line because the file it was asked for is open in a spreadsheet, which on Windows
    holds a lock that refuses the write. Opening for update touches nothing and creates nothing.
    """
    path = Path(path)
    if not path.parent.exists():
        return f"there is no folder {path.parent}"
    if path.exists():
        try:
            open(path, "r+b").close()
        except OSError as problem:
            return f"{path} cannot be written ({problem.strerror or problem}), so it may be open elsewhere"
    return ""

def read_prices(path, column="Adj Close") -> pd.Series:
    """A price series from a csv written by xdownload_prices.py."""
    return pd.read_csv(path, index_col="Date", parse_dates=["Date"])[column].sort_index()

def returns_on(prices: pd.Series, dates) -> pd.Series:
    """Returns between consecutive dates, using the last price on or before each one, so they cover
    the same intervals as the futures returns even when the two markets keep different holidays.

    The fill stops where the prices do. Carrying the last price past the end of a stale file does not
    show up as missing data but as a run of days on which the market did not move, and those days sit
    in the betas and the overlays as if they were real. Missing is the truth, and the join drops it.
    """
    wanted = pd.DatetimeIndex(dates)
    filled = prices.reindex(prices.index.union(wanted)).ffill().reindex(wanted)
    return filled.where(wanted <= prices.index.max()).pct_change(fill_method=None)

def rate_returns(path, dates) -> pd.Series:
    """Daily risk-free returns from a csv of 13 week Treasury bill yields, or zero without one.

    A futures position needs no funding, so its returns are already in excess of the rate; taking the
    rate off SPY puts the two on the same footing.
    """
    if path and Path(path).exists():
        yields = read_prices(path, "Close") / 100 / TRADING_DAYS
        return yields.reindex(yields.index.union(dates)).ffill().reindex(dates).fillna(0)
    return pd.Series(0.0, index=pd.DatetimeIndex(dates))

def compound_to(r: pd.Series, rule) -> pd.Series:
    """Daily returns compounded into whole calendar weeks, months, quarters or years.

    The first and last of them are whatever part the data cover, which over twenty years moves nothing
    worth correcting for at the shorter frequencies. At the annual one it is a larger share of a small
    sample, which is a reason to read that row loosely rather than to adjust it.
    """
    return r if rule is None else (1 + r).groupby(pd.Grouper(freq=rule)).prod() - 1

def period_ends(r: pd.Series, rule) -> pd.Series:
    """The last trade date inside each compounded period, labelled as compound_to labels them.

    The label pandas gives a group is the calendar end of the period, which is often not a trading day
    and, for the period the data stop in, has not arrived: an annual row would otherwise date its worst
    year to a December that is still months away. The last date actually in the group is the one that
    can be looked up in the contract file.
    """
    if rule is None:
        return pd.Series(r.index, index=r.index)
    ends = r.groupby(pd.Grouper(freq=rule)).apply(lambda part: part.index.max())
    return ends.dropna()

def to_monthly(r: pd.Series) -> pd.Series:
    """Daily returns compounded into whole calendar months."""
    return compound_to(r, "ME")

def return_over_drawdown(equity: pd.Series, days: int, periods: float = TRADING_DAYS) -> float:
    """Annual return divided by the worst fall from a high, the ratio sometimes called MAR.

    It says how much a year of holding earned for each point of the deepest hole it had to sit through,
    which is the question a hedge is meant to answer. Read it only against series of the same length:
    the longer a history runs, the deeper its worst drawdown is likely to be, so a short sample
    flatters itself.
    """
    drawdown = (equity / equity.cummax() - 1).min()
    annual = equity.iloc[-1] ** (periods / days) - 1
    return annual / abs(drawdown) if drawdown < 0 else np.nan

def performance(returns: pd.Series, rate: pd.Series, period: str = "Daily") -> dict:
    """What the position did: growth, risk, the shape of the returns and the worst stretches.

    The daily returns are compounded into the period named before anything is measured. Nothing about
    the position changes, only the ruler, and the ruler matters here: daily returns are far from
    independent and far from normal, so a deviation annualized by the root of 252 is not the deviation
    longer periods show, and the skew and kurtosis of single days say little about the months a holder
    is actually shown. The growth is the same however it is cut, so Ann_Return barely moves down the
    rows and stands as a check that they all describe one position.

    The longer rows are worth less as statistics than they look. Twenty years is about a thousand
    weeks but only twenty odd years, so the annual row's deviation, skew and kurtosis rest on a handful
    of numbers, and its drawdown, taken from year ends alone, misses everything that happened inside a
    year. Read the bottom rows for the size of a typical period and the worst one, not for their
    moments.

    Worst and Best are the worst and best of whatever period the row measures, so a day on the daily
    row and a calendar year on the annual one, and the two dates beside them are the last trade date
    inside each of those periods.
    """
    rule, periods = PERIODS[period]
    ends = period_ends(returns, rule)
    returns, rate = compound_to(returns, rule), compound_to(rate, rule)
    returns, rate = returns.reindex(ends.index), rate.reindex(ends.index)
    equity = (1 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1
    return {
        "Ann_Return": equity.iloc[-1] ** (periods / len(returns)) - 1,
        "Ann_Mean": returns.mean() * periods,
        "Ann_Vol": returns.std() * np.sqrt(periods),
        "Sharpe": (returns - rate).mean() / returns.std() * np.sqrt(periods),
        "Skew": returns.skew(),
        "Kurtosis": returns.kurt(),
        "Max_Drawdown": drawdown.min(),
        "Return_Over_DD": return_over_drawdown(equity, len(returns), periods),
        "Worst": returns.min(),
        "Best": returns.max(),
        "Worst_Period": str(ends[returns.idxmin()].date()),
        "Best_Period": str(ends[returns.idxmax()].date()),
    }

def carry_by_tenor(carry: pd.DataFrame, tenors: pd.DataFrame, first: int, last: int) -> pd.DataFrame:
    """The carry per day of whichever contract stood at each tenor, one column per tenor.

    At most one contract holds a tenor on a day, so averaging across the row picks that one out.
    """
    aligned = tenors.reindex(index=carry.index, columns=carry.columns)
    return pd.DataFrame({tenor: carry.where(aligned.eq(tenor)).mean(axis=1)
                         for tenor in range(first, last + 1)})

def cost_of_the_protection(parts: pd.DataFrame, vix: pd.Series, short: bool = False,
                           tenor_carry: pd.DataFrame = None, footer: dict = None) -> pd.DataFrame:
    """What holding this has cost to carry, over the years and right now.

    Carry is what the position earns in a day if the curve does not move: each contract slides one day
    along the curve towards the next. Held long it is negative most of the time, and that is the price
    of the protection; what is left of the return is the curve moving, which is the payout.

    Carry grows with the level of VIX, so the same points mean different things at VIX 12 and VIX 40.
    The second column divides by VIX and the percentile is taken on that, which is what says whether
    the protection is dear by its own standards. The percentile counts the days at or below today, so
    it runs from 0 on the dearest day this has ever been to 100 on the cheapest. Everything is
    annualized.

    Sold short the position collects that carry instead of paying it, so the extremes swap places and
    are labelled accordingly.

    Ck is the carry the position earns at tenor k, signed as it holds the contract rather than as the
    contract itself carries, so that it reads against Carry_Ann beside it. It says where the cost
    comes from: it
    decays roughly as one over the tenor, so the front of the curve is where a long position bleeds.
    The two aggregate rows are each column's own average and median, but the extremes and today are the
    day named in the row, so those rows read across as the carry curve on that one day. The position's
    own carry then sits between its contracts', its weights being shares of one, on every day of this
    history but one where the curve carries a figure at every tenor and the whole book is inside the
    strip. Where it does not, the book is holding a contract from outside the strip, or a tenor has no
    carry that day, so the columns shown are not the whole of what was held.

    footer holds rows of text keyed by column name, used for the price and the name of the contract at
    each tenor on the last trade date. They go under the current row rather than over the headings
    because the contract at a tenor is a different one on every row: the fourth was VXZ12 on the
    dearest day of this history and VXN20 on the cheapest, and the two aggregate rows span thousands of
    days, where no contract stands at any tenor at all.
    """
    carry = parts["Carry"] * TRADING_DAYS
    relative = (parts["Carry"] / vix.reindex(parts.index)).dropna() * TRADING_DAYS
    latest = carry.index[-1]
    least, most = ("Least earned", "Most earned") if short else ("Dearest", "Cheapest")
    rows = {"Average": {"Carry_Ann": carry.mean(), "Carry_Over_VIX": relative.mean()},
            "Median": {"Carry_Ann": carry.median(), "Carry_Over_VIX": relative.median()},
            least: {"Carry_Ann": carry.min(), "Carry_Over_VIX": relative.min()},
            most: {"Carry_Ann": carry.max(), "Carry_Over_VIX": relative.max()},
            f"Now ({latest.date()})": {"Carry_Ann": carry.iloc[-1], "Carry_Over_VIX": relative.iloc[-1]}}
    if tenor_carry is not None:
        # the aggregate rows are each column's own; the rest are the day named, so a row is one curve
        annual = tenor_carry.reindex(parts.index) * TRADING_DAYS
        days = {least: carry.idxmin(), most: carry.idxmax(), f"Now ({latest.date()})": latest}
        for tenor in annual.columns:
            rows["Average"][f"C{tenor}"] = annual[tenor].mean()
            rows["Median"][f"C{tenor}"] = annual[tenor].median()
            for label, day in days.items():
                rows[label][f"C{tenor}"] = annual[tenor].get(day, np.nan)
    frame = pd.DataFrame(rows).T
    frame["Percentile"] = np.nan
    frame.loc[f"Now ({latest.date()})", "Percentile"] = (relative <= relative.iloc[-1]).mean() * 100
    if footer:
        # these belong to the day the row above names, not to the columns: every other row is a
        # different day, and the two aggregate rows are thousands of them, so they go here
        frame = frame.map(lambda value: f"{value:.4f}" if pd.notna(value) else "")
        for label, values in footer.items():
            frame.loc[label] = [values.get(column, "") for column in frame.columns]
    return frame

def equity_betas(returns: pd.Series, excess_spy: pd.Series, in_points: bool = False) -> pd.DataFrame:
    """Beta to SPY over all days, and separately on the days SPY fell and rose.

    A hedge is worth more when it moves further against a falling market than it gives up in a rising
    one, so the two halves are worth seeing apart.

    The means and standard deviations of both sides are annualized. On the down and up rows those are
    conditional averages put on a yearly scale for reading, not what such a year would look like: the
    market does not fall every day.

    With in_points the position is read in VIX points instead of as a fraction of what it is worth, and
    the market in per cent, so the beta is the points the position moves for each per cent the market
    moves. That is the number to size a hedge with: it does not drift as the level of the curve does,
    whereas a beta between two percentage returns divides by a price that ranged from 12 to 50 here.
    """
    if in_points:
        excess_spy = excess_spy * 100
    d = pd.concat([returns.rename("r"), excess_spy.rename("m")], axis=1).dropna()
    rows = {}
    for name, part in (("all days", d), ("SPY down", d[d["m"] < 0]), ("SPY up", d[d["m"] > 0])):
        covariance = np.cov(part["r"], part["m"])
        rows[name] = {"Days": len(part), "Beta": covariance[0, 1] / covariance[1, 1],
                      "Correlation": part["r"].corr(part["m"]),
                      "Position_Mean": part["r"].mean() * TRADING_DAYS,
                      "Position_SD": part["r"].std() * np.sqrt(TRADING_DAYS),
                      "SPY_Mean": part["m"].mean() * TRADING_DAYS,
                      "SPY_SD": part["m"].std() * np.sqrt(TRADING_DAYS)}
    return pd.DataFrame(rows).T.astype({"Days": int})

def shape_models(max_power: int = 2, down_powers: bool = False) -> dict:
    """The shapes to fit: a straight line, a kink at zero, and curves up to max_power.

    The line is the baseline. Adding the size of the move is the same as letting the slope differ
    either side of zero, a kink rather than a curve. The powers bend it: a square bends it the same way
    on both sides, a cube bends it one way on the way down and the other on the way up, and each higher
    power fits the far tails harder, which is where the fewest days are. Every shape contains the
    straight line, so what is being tested is whether the extra terms earn their place; the criteria
    say whether they do.

    With down_powers the powers are also fitted switched off above zero, so they bend the line only
    when the market falls. That is the shape a hedge is supposed to have, and for this position it has
    not been the shape the criteria prefer, so it is left out unless asked for. Note that the other
    shapes are not symmetric either: the size of the move is a kink, with a different slope each side
    of zero, and an odd power bends one way down and the other way up.
    """
    models = {"linear": lambda x: {"SPY": x},
              "linear + |SPY|": lambda x: {"SPY": x, "|SPY|": np.abs(x)}}
    for power in range(2, max_power + 1):
        powers = tuple(range(2, power + 1))
        models[f"powers to {power}"] = (
            lambda x, powers=powers: {"SPY": x} | {f"SPY^{p}": x ** p for p in powers})
        if down_powers:
            models[f"down powers to {power}"] = (
                lambda x, powers=powers: {"SPY": x} | {f"down SPY^{p}": np.where(x < 0, x ** p, 0.0)
                                                       for p in powers})
    return models

def overlapping_days(a: pd.Series, b: pd.Series) -> int:
    """How many days the two series both have a value for, which is what a regression on them uses."""
    return len(pd.concat([a, b], axis=1).dropna())

def fit_shape(y: np.ndarray, terms: dict) -> dict:
    """Least squares with robust standard errors, and the usual information criteria.

    The t statistics use the White correction, since daily returns are far from equally variable. The
    criteria come from the Gaussian likelihood, which these fat tailed residuals do not obey: read them
    as a ranking of fit against the number of terms, and treat small differences as no difference.
    """
    names = ["const"] + list(terms)
    design = np.column_stack([np.ones_like(y)] + [np.asarray(v, dtype=float) for v in terms.values()])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    residuals = y - design @ coefficients
    n, k = design.shape
    inverse = np.linalg.pinv(design.T @ design)
    meat = design.T @ (design * (residuals ** 2)[:, None])
    covariance = inverse @ meat @ inverse * n / (n - k)  # White, with the small sample correction
    errors = np.sqrt(np.diag(covariance))
    rss = residuals @ residuals
    log_likelihood = -0.5 * n * (np.log(2 * np.pi) + np.log(rss / n) + 1)
    return {"names": names, "coefficients": coefficients, "t": coefficients / errors,
            "R2": 1 - rss / ((y - y.mean()) ** 2).sum(),
            "AIC": -2 * log_likelihood + 2 * k, "BIC": -2 * log_likelihood + k * np.log(n)}

def strip_points(daily: pd.DataFrame, prices: pd.DataFrame) -> pd.Series:
    """The day's gain of the position in VIX points, rather than as a fraction of what it is worth.

    The weights held from the previous close, times what each of those contracts moved in points. It is
    the same position read in the units the curve is quoted in, which do not shrink as the price falls.
    """
    held = expand_positions(daily).shift(1)
    moves = prices.reindex(index=held.index, columns=held.columns).diff()
    return (held * moves).sum(axis=1)

def shape_table(returns: pd.Series, spy: pd.Series, y_scale: float = 100, max_power: int = 2,
                down_powers: bool = False) -> pd.DataFrame:
    """How the position's returns line up with the market's, under four shapes.

    Both sides are in per cent, so a coefficient reads against a one per cent move. The models are the
    straight line; the line plus the size of the move, which is the same as letting the slope differ
    either side of zero; the line plus the square of the move, which bends it the same way on both
    sides; and the line plus the square only when the market falls, which bends it only on the way
    down. Each contains the straight line, so the extra term is what is being tested, but none of the
    three contains another, which is what the criteria are for: lower is better.

    AIC_rank puts the shapes in order, 1 for the one the criterion likes best.

    y_scale turns the position's side into the units wanted: 100 for a percentage return, 1 for a gain
    already counted in VIX points. The criteria compare shapes fitted to the same left hand side, not
    one left hand side against another, and an R squared belongs to the series it was fitted to, so the
    two tables say which shape suits each reading, not which reading is true.
    """
    d = pd.concat([returns.rename("y") * y_scale, spy.rename("x") * 100], axis=1).dropna()
    y, x = d["y"].to_numpy(), d["x"].to_numpy()
    rows = {}
    for name, build in shape_models(max_power, down_powers).items():
        fit = fit_shape(y, build(x))
        row = {}
        for term, coefficient, t in zip(fit["names"], fit["coefficients"], fit["t"]):
            row[term] = coefficient
            if term not in ("const", "SPY"):  # the terms being tested carry their t, the line's do not
                row[f"t({term})"] = t
        row |= {"R2": fit["R2"], "AIC": fit["AIC"], "BIC": fit["BIC"]}
        rows[name] = row
    frame = pd.DataFrame(rows).T
    frame.insert(0, "AIC_rank", frame["AIC"].rank().astype(int))  # 1 is the shape the criterion likes
    frame.index.name = "Shape"
    return frame

def shape_payoffs(returns: pd.Series, spy: pd.Series,
                  moves=(-20, -15, -10, -5, -3, -1, 1, 3, 5, 10, 15, 20),
                  max_power: int = 2, down_powers: bool = False) -> pd.DataFrame:
    """What each shape says the position returns on a given move in the market, in per cent.

    The coefficients are hard to weigh against each other; this is the same models read out as the
    thing a holder cares about, and it shows where they disagree, which is in the tails.

    The far columns are beyond anything the market has done in a day here, so those are the shapes
    extrapolated rather than fitted, and the higher the power the wilder the extrapolation. Read them
    as what each shape implies, not as a forecast; the line above the table says how far the market
    actually moved.
    """
    d = pd.concat([returns.rename("y"), spy.rename("x")], axis=1).dropna() * 100
    y, x = d["y"].to_numpy(), d["x"].to_numpy()
    rows = {}
    for name, build in shape_models(max_power, down_powers).items():
        fit = fit_shape(y, build(x))
        moves_array = np.array(moves, dtype=float)
        design = np.column_stack([np.ones_like(moves_array)]
                                 + [np.asarray(v, dtype=float) for v in build(moves_array).values()])
        rows[name] = dict(zip([f"SPY {move:+g}%" for move in moves], design @ fit["coefficients"]))
    frame = pd.DataFrame(rows).T
    frame.index.name = "Shape"
    return frame

def move_buckets(returns: pd.Series, spy: pd.Series, edges=(-5, -3, -1, 0, 1, 3, 5),
                 y_scale: float = 100) -> pd.DataFrame:
    """What the position actually did, by how far the market moved, without fitting anything.

    Each row is the days whose market move fell in that band, with what the market and the position
    averaged, how much the position varied within the band, and how closely the two moved together
    there. The market is in per cent; y_scale leaves the position in the units wanted, 100 for a
    percentage return and 1 for a gain already counted in VIX points.

    The first row is every day together, to read the bands against; on that row the ratio and the beta
    are the ones the tables above report.

    This is the companion to the fitted shapes: it says nothing beyond the days themselves, so it does
    not extrapolate, and the far rows hold few days, which the count says plainly. Ratio is the
    position's average divided by the market's, with the sign taken off, so it reads as how much the
    position moved for each per cent of the market's move. Beta is fitted within the band alone, so it
    says how the position tracked the market's wobbles inside that range, which is a different thing
    from the ratio of the two averages and much the noisier of the pair: a narrow band leaves little
    for it to work with.
    """
    def summarize(part):
        return {"Days": len(part), "SPY_Mean": part["x"].mean(),
                "Position_Mean": part["y"].mean(), "Position_SD": part["y"].std(),
                "Ratio": abs(part["y"].mean() / part["x"].mean()) if part["x"].mean() else np.nan,
                "Correlation": part["y"].corr(part["x"]),
                "Beta": (np.cov(part["y"], part["x"])[0, 1] / part["x"].var()
                         if len(part) > 2 and part["x"].var() > 0 else np.nan)}

    d = pd.concat([returns.rename("y") * y_scale, spy.rename("x") * 100], axis=1).dropna()
    bands = pd.cut(d["x"], [-np.inf, *edges, np.inf])
    rows = {"all days": summarize(d)}  # the whole sample first, to read the bands against
    for band, part in d.groupby(bands, observed=True):
        rows[str(band)] = summarize(part)
    frame = pd.DataFrame(rows).T
    frame.index.name = "SPY move, per cent"
    return frame.astype({"Days": int})

def adjusted_sharpe(r: pd.Series, rate: pd.Series, periods: float) -> float:
    """The Sharpe ratio of Pezier and White (2006), penalized for negative skew and fat tails.

        ASR = SR * (1 + (S / 6) * SR - (K / 24) * SR ** 2)

    with S the skew and K the excess kurtosis, all taken at the frequency of the returns given, as the
    expansion requires, and annualized at the end like the ratio it corrects.

    It is a third order expansion, so it means something only for mild departures from normality. Daily
    returns here are not mild, with excess kurtosis from 15 to 26, but the per day ratio is so small
    that its square leaves the correction near nothing: the daily column will sit on top of the Sharpe
    it adjusts. Monthly returns are the ones the adjustment is worth reading, being closer to normal
    and carrying a ratio large enough for the skew term to bite.
    """
    sharpe = (r - rate).mean() / r.std()
    adjusted = sharpe * (1 + r.skew() / 6 * sharpe - r.kurt() / 24 * sharpe ** 2)
    return adjusted * np.sqrt(periods)

def held_stats(r: pd.Series, rate: pd.Series, monthly: bool = False) -> dict:
    """What a held position did, from either daily or monthly returns.

    The frequency is not a presentation choice. Daily returns here are negatively autocorrelated, at
    about -0.10 for SPY with a slice on top, so a day's move is partly given back the next; annualizing
    a daily deviation by the root of 252 ignores that and puts SPY's volatility at 19.5% against the
    15.1% its monthly returns show. Skew moves further still: the hedge pays on the day a market
    breaks, which daily returns reward, while over a whole month the equity loss and the carry catch up
    with it. Drawdown taken from month ends misses what happened inside a month, so it is the kinder of
    the two numbers.
    """
    periods = 12 if monthly else TRADING_DAYS
    equity = (1 + r).cumprod()
    return {"Ann_Return": equity.iloc[-1] ** (periods / len(r)) - 1,
            "Ann_Vol": r.std() * np.sqrt(periods),
            "Sharpe": (r - rate).mean() / r.std() * np.sqrt(periods),
            "Sharpe_adj": adjusted_sharpe(r, rate, periods),
            "Max_Drawdown": (equity / equity.cummax() - 1).min(),
            "Return_Over_DD": return_over_drawdown(equity, len(r), periods),
            "Skew": r.skew(),
            "Worst_Month": r.min() if monthly
            else ((1 + r).rolling(21).apply(np.prod, raw=True) - 1).min()}

def overlay_table(returns: pd.Series, spy: pd.Series, rate: pd.Series,
                  sizes=(0, 0.05, 0.10, 0.20, 0.30), monthly: bool = False) -> pd.DataFrame:
    """SPY held in full plus a slice of this position on top.

    Futures need no capital, so the slice is added to a full holding of SPY rather than funded by
    selling part of it. A slice of zero is SPY on its own, which the caller puts at the front so there
    is always something to read the rest against; a negative slice sells the position short instead,
    which is the short volatility trade rather than a hedge.

    With monthly the slice is still held and rebalanced daily, as it must be; only the returns are
    compounded into months before being measured, which is how a holder would be shown them.
    """
    rate = to_monthly(rate) if monthly else rate
    rows = {}
    for size in sizes:
        r = spy + size * returns
        rows[f"{size:.0%}"] = held_stats(to_monthly(r) if monthly else r, rate, monthly)
    frame = pd.DataFrame(rows).T
    frame.index.name = "Slice of SPY"
    return frame

def scaled_overlay_table(returns: pd.Series, spy: pd.Series, rate: pd.Series, vix: pd.Series,
                         powers=(0, 0.5, 1.0, 1.5), target: float = 0.15, floor: float = 0.0,
                         cap: float = 1.0, monthly: bool = False) -> pd.DataFrame:
    """The same slice, but varying with the level of VIX instead of staying fixed.

    SPY's own volatility rises roughly one for one with VIX while this position's rises about half as
    fast, so a fixed slice covers less and less equity risk as VIX rises; holding the hedge ratio
    steady takes a power near 0.5. Larger powers put more on when volatility is already high, which is
    a bet on crises repeating rather than a risk adjustment. The size uses the previous close's VIX
    over its own average up to the day before, and averages out near target whatever the power, so the
    rows can be read against each other. A power of zero is the slice held constant, which the caller
    puts at the front so there is always something to compare with.

    floor and cap hold the slice between them. A floor matters because the rule only grows the position
    after VIX has risen, so a fall that starts from a quiet market finds the hedge at its smallest; a
    cap matters because the unheld rules reach two thirds of the equity notional in a crisis, which is
    a great deal of margin to find on the day. Holding the slice between them moves its average, so the
    rows are then no longer at quite the same size: the Mean_Size column says what each one came to.
    """
    monthly_rate = to_monthly(rate) if monthly else rate
    rows = {}
    for power in powers:
        raised = vix.shift(1) ** power
        size = (target * raised / raised.expanding(TRADING_DAYS).mean()).clip(floor, cap).fillna(target)
        r = spy + size * returns
        stats = held_stats(to_monthly(r) if monthly else r, monthly_rate, monthly)
        del stats["Skew"]
        rows["Constant" if power == 0 else f"VIX^{power:g}"] = stats | {
            "Mean_Size": size.mean(),
            "Median_Size": size.median(),
            "Size_Range": f"{size.min():.0%}-{size.max():.0%}",
        }
    frame = pd.DataFrame(rows).T
    frame.index.name = "Slice sized by"
    return frame

def compare_with_fund(returns: pd.Series, fund_path, fee: float, name: str) -> str:
    """How closely this matches the fund that holds the same index, from the fund's own prices.

    Only the two strips a fund actually holds can be checked this way, which is what FUNDS lists.
    Comparing any other strip with one of them measures the difference between two positions rather
    than the construction of either.

    Pass one unit of the index held long, whatever the position actually is: this asks whether the
    construction is right, and a short or a multiple of it would only put its own sign and size on the
    answer.

    Pass the returns before trading costs: the index itself has none, and what separates the fund from
    the index is its fee and its own trading, so the difference should come out near the fee.
    """
    if not (fund_path and Path(fund_path).exists()):
        return (f"no {name} prices at {fund_path}, so no comparison; "
                f'run "python xdownload_prices.py {name}" to fetch them')
    fund = read_prices(fund_path).pct_change()
    dates = returns.index.intersection(fund.index)
    here, there = returns.reindex(dates), fund.reindex(dates)
    both = here.notna() & there.notna() & (here != 0)
    here, there = here[both], there[both]
    difference = (there - here).mean() * TRADING_DAYS
    return (f"against {name} itself over {len(here)} days from {here.index[0].date()}: "
            f"correlation {np.corrcoef(here, there)[0, 1]:.3f}, "
            f"this {here.mean() * TRADING_DAYS:+.1%} a year at {here.std() * np.sqrt(TRADING_DAYS):.1%} "
            f"volatility against the fund's {there.mean() * TRADING_DAYS:+.1%} at "
            f"{there.std() * np.sqrt(TRADING_DAYS):.1%}, a difference of {difference:+.2%} a year "
            f"against a fee of {fee:.2%}")

def plot(returns: pd.Series, spy: pd.Series, overlays: dict, path, years: str) -> None:
    """A chart of what a dollar became, and of how deep the holes were."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = {"SPY": spy, "long mid term VIX futures": returns} | overlays
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
    figure.patch.set_facecolor(SURFACE)
    for panel in axes:
        panel.set_facecolor(SURFACE)
        panel.grid(True, color="#e6e5e1", linewidth=1)
        panel.set_axisbelow(True)
        for side in ("top", "right"):
            panel.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            panel.spines[side].set_color("#d8d7d2")
        panel.tick_params(colors=MUTED, labelsize=8)
    for slot, (name, r) in enumerate(series.items()):
        growth = (1 + r.fillna(0)).cumprod().clip(lower=1e-4)
        drawdown = growth / growth.cummax() - 1
        colour = PALETTE[slot % len(PALETTE)]
        axes[0].plot(growth.index, growth.to_numpy(), color=colour, linewidth=2, label=name)
        axes[1].plot(drawdown.index, drawdown.to_numpy() * 100, color=colour, linewidth=2, label=name)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Growth of $1", color=MUTED, fontsize=9)
    axes[0].set_title(f"Long mid term VIX futures, {years}", color=INK, fontsize=13, loc="left", pad=14)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=MUTED, loc="upper left")
    axes[1].set_ylabel("Drawdown, per cent", color=MUTED, fontsize=9)
    axes[1].set_title("How far below its own high water mark", color=INK, fontsize=11, loc="left")
    figure.tight_layout()
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    print(f"\nwrote the chart to {path}")

def report(first: int, last: int, df: pd.DataFrame, prices: pd.DataFrame, vix_close: pd.Series,
           spy_prices: pd.Series, args, sweeping: bool = False) -> None:
    """Everything the program has to say about one position, printed.

    The pair names the tenors held: first below last is the S&P style index rolled through the range,
    and a pair of equal tenors is that one contract rolled on each settlement day. The contract file,
    the grid of prices, VIX and SPY are read once by the caller and passed in, so a sweep over several
    positions pays for the reading once rather than per position.

    With sweeping the files asked for by --plot and --output get the tenors put in their names, since
    otherwise each run of a sweep would write over the one before it.
    """
    single = first == last  # one tenor is the contract held alone, rolled at each settlement
    if single:
        daily = backtest_single_contract(df, first, int(np.sign(args.scale)), abs(args.scale),
                                         args.cost)
    else:
        daily = backtest_index(df, first, last, int(np.sign(args.scale)),
                               abs(args.scale), args.cost, rebalance_days=args.rebalance_days)
    active = daily.index[daily["Gross_Leverage"] > 0]
    daily = daily.loc[daily.index >= active[0]]
    returns = daily["Net_Return"]
    vix = vix_close.reindex(daily.index)
    rate = rate_returns(args.rf_file, daily.index)
    spy = returns_on(spy_prices, daily.index)
    frame = pd.concat([returns.rename("strategy"), spy.rename("spy"), rate.rename("rate"),
                       vix.rename("vix")], axis=1).dropna()

    # the program is written for the long side, where the position is a hedge paying away its carry;
    # sold short it is the opposite trade in every respect, so the commentary follows the sign
    short = args.scale < 0
    side = "short" if short else "long"
    if single:
        print(f"{side} one contract, the {ordinal(first)} out, rolled on each settlement "
              f"day, at {abs(args.scale):g} times, costing {args.cost:g} VIX points a contract; its "
              f"maturity falls through the month, so it averages about half a month less than "
              f"{first} month{'s' if first > 1 else ''}")
    else:
        rolling = ("rolled a little each day" if args.rebalance_days == 1
                   else f"brought back to the index every {args.rebalance_days} trading days")
        print(f"{side} the contracts {first} to {last} months out, {rolling}, at "
              f"{abs(args.scale):g} times the index, costing {args.cost:g} VIX points a contract")
    # the span every table uses, which is where the futures, VIX and SPY all have a figure
    covered = daily.loc[frame.index]
    print(f"{len(covered)} trading days from {covered.index[0].date()} to {covered.index[-1].date()}, "
          f"average maturity {covered['Maturity_Days'].mean():.0f} calendar days, "
          f"turnover {covered['Turnover'].mean() * TRADING_DAYS:.1f} times a year, "
          f"traded on {(covered['Turnover'] > 0).mean() * TRADING_DAYS:.0f} days a year")
    # a file running ahead of the futures costs nothing; one running behind cuts the tables short
    behind = {name: when for name, when in (("VIX", vix_close.index[-1]), ("SPY", spy_prices.index[-1]))
              if when < daily.index[-1]}
    if behind and frame.index[-1] < daily.index[-1]:
        print(f"the futures reach {daily.index[-1].date()} but the tables stop at "
              f"{frame.index[-1].date()}, because "
              + " and ".join(f"{name} ends {when.date()}" for name, when in behind.items())
              + ". Run python xupdate_data.py to bring the price files level")
    # the fund holds one unit of the index long, so the check is of the construction, not of the
    # multiple or the side held: take the scale back out before comparing
    # only a strip a fund holds can be checked, and only when it is rolled the way the index is
    fund = FUNDS.get((first, last)) if args.rebalance_days == 1 else None
    if fund:
        name, option, fee = fund
        print(compare_with_fund(daily["Gross_Return"] / (args.scale or 1), getattr(args, option),
                                fee, name))
    else:
        print("no fund holds this strip, so there is nothing to check the construction against")

    with pd.option_context("display.width", 200, "display.max_columns", None,
                           "display.float_format", "{:.4f}".format):
        own = pd.DataFrame({name: performance(frame["strategy"], frame["rate"], name)
                            for name in PERIODS}).T
        own.index.name = side.title()
        print(f"\nstand-alone performance\n{own}")
        carry, grid = carry_by_contract(df, vix_close), tenor_grid(df)
        parts = split_return(expand_positions(daily), carry, daily["Gross_Return"]).loc[frame.index]
        detail = (position_detail(daily, prices, carry, grid, last)
                  if args.output or args.best_days or args.worst_days else None)
        # as of the previous close, which is when the position and its carry were both set, and signed
        # the way the position holds them: sold short, a contract in contango earns its carry
        tenor_carry = carry_by_tenor(carry, grid, first, last).shift(1) * (np.sign(args.scale) or 1)
        carried = ("what the position earns in carry, and what the curve took back" if short
                   else "what the protection costs to carry, and what the curve paid back")
        # what the curve holds at each tenor on the last trade date, for the rows under the table
        standing = grid.reindex(parts.index).iloc[-1].dropna()
        held_now = {int(tenor): expiry for expiry, tenor in standing.items() if first <= tenor <= last}
        closes = prices.reindex(parts.index).iloc[-1]
        footer = {"Price now": {f"C{tenor}": f"{closes[expiry]:.4f}" for tenor, expiry in held_now.items()
                                if pd.notna(closes.get(expiry))},
                  "Contract now": {f"C{tenor}": contract_symbol(expiry)
                                   for tenor, expiry in held_now.items()}}
        print(f"\n{carried}, annualized; Ck is the carry the position earns at tenor k, and the extreme "
              f"and current rows read across as the carry curve on that day\n"
              f"{cost_of_the_protection(parts, vix_close, short, tenor_carry, footer)}")
        by_year = parts.groupby(parts.index.year).sum()
        by_year.index.name = "Year"
        print(f"\ncarry and curve by year, added up day by day rather than compounded\n{by_year.T}")
        ends = ("earning least", "earning most") if short else ("dearest", "cheapest")
        weigh = ("a premium to collect, not a forecast of what the short will give back" if short else
                 "a cost to weigh, not a forecast of what the protection will pay")
        print(f"the percentile counts the days at or below today, so it runs from 0 on the day this was "
              f"{ends[0]} to 100 on the day it was {ends[1]}; over this history a dearer carry has not "
              f"bought a larger payout, so it is {weigh}")
        for count, worst in ((args.worst_days, True), (args.best_days, False)):
            if count:
                which = "worst" if worst else "best"
                print(f"\nthe {count} {which} days, with what was held and what each contract did; Pk is "
                      f"the close at tenor k, Wk the weight held there and Rk that contract's own "
                      f"return\n"
                      f"{extreme_days(daily, parts, frame, detail, first, last, count, worst)}")
        print(f"\nagainst SPY in excess of the rate\n"
              f"{equity_betas(frame['strategy'], frame['spy'] - frame['rate'])}")
        points = strip_points(daily, prices).reindex(frame.index)
        sizing = "size the exposure with" if short else "size a hedge with"
        print(f"\nthe same in VIX points for each per cent the market moves, which is what to {sizing}; "
              f"the position's mean and deviation are then in points a year\n"
              f"{equity_betas(points, frame['spy'] - frame['rate'], in_points=True)}")
        if not args.noreg:
            excess = frame["spy"] - frame["rate"]
            print(f"\nthe shape of the two together over "
                  f"{overlapping_days(frame['strategy'], excess)} days, both in per cent, so a "
                  f"coefficient reads against a one per cent move; lower AIC and BIC are better\n"
                  f"{shape_table(frame['strategy'], excess, max_power=args.shape_power, down_powers=args.down_powers)}")

            print(f"\nthe same shapes over {overlapping_days(points, excess)} days with the position "
                  f"read in VIX points rather than per cent; an R "
                  f"squared belongs to the series it was fitted to, so this says which reading of the "
                  f"position moves more tidily with the market, not which is true\n"
                  f"{shape_table(points, excess, y_scale=1, max_power=args.shape_power, down_powers=args.down_powers)}")
            worst, best = excess.min() * 100, excess.max() * 100
            print(f"\nthe same models read out as what they say the position returns, in per cent; the "
                  f"market's own worst and best days here were {worst:.1f}% and {best:.1f}%, so the "
                  f"columns beyond those are the shapes extrapolated\n"
                  f"{shape_payoffs(frame['strategy'], excess, max_power=args.shape_power, down_powers=args.down_powers)}")
            quiet = ~frame.index.to_period("M").isin(pd.PeriodIndex(["2018-02", "2020-03"], freq="M"))
            print(f"\nand without February 2018 and March 2020, where the bends come from\n"
                  f"{shape_payoffs(frame['strategy'][quiet], excess[quiet], max_power=args.shape_power, down_powers=args.down_powers)}")
            print(f"\nand what the position actually did, by how far the market moved, without fitting "
                  f"anything\n{move_buckets(frame['strategy'], excess)}")
            print(f"\nthe same bands with the position in VIX points, where a fall in the level of the "
                  f"curve does not shrink what a point is worth\n"
                  f"{move_buckets(points, excess, y_scale=1)}")
        scaled = dict(powers=[0] + list(args.overlay_powers), target=args.overlay_size,
                      floor=args.min_slice, cap=args.max_slice)
        sizes = [0] + [size for size in args.slices if size != 0]
        print(f"\nSPY held in full, with a slice of it on top\n"
              f"{overlay_table(frame['strategy'], frame['spy'], frame['rate'], sizes)}")
        print(f"\nthe same, measured on whole calendar months, which is how returns are usually "
              f"reported; Worst_Month is then the worst single month rather than the worst 21 days, and "
              f"the drawdown only sees month ends\n"
              f"{overlay_table(frame['strategy'], frame['spy'], frame['rate'], sizes, monthly=True)}")
        print(f"\nthe same slice, sized by the level of VIX\n"
              f"{scaled_overlay_table(frame['strategy'], frame['spy'], frame['rate'], frame['vix'], **scaled)}")
        print(f"\nthe same, measured on whole calendar months\n"
              f"{scaled_overlay_table(frame['strategy'], frame['spy'], frame['rate'], frame['vix'], monthly=True, **scaled)}")
        yearly = frame.groupby(frame.index.year).apply(
            lambda year: pd.Series({"Strategy": (1 + year["strategy"]).prod() - 1,
                                    "SPY": (1 + year["spy"]).prod() - 1,
                                    "Beta": np.cov(year["strategy"], year["spy"])[0, 1] / year["spy"].var()}))
        yearly.index.name = "Year"
        print(f"\nby year\n{yearly.T}")

    print(f"\n{closing_note(frame, args.overlay_size)}")
    if args.plot:
        chart = suffixed(args.plot, first, last) if sweeping else args.plot
        overlays = {f"SPY plus {size:.0%}": frame["spy"] + size * frame["strategy"] for size in (0.1,)}
        plot(frame["strategy"], frame["spy"], overlays, chart,
             f"{frame.index[0].year} to {frame.index[-1].year}")
    if args.output:
        out = suffixed(args.output, first, last) if sweeping else args.output
        written = daily.drop(columns="Positions", errors="ignore").join(detail)
        written.to_csv(out)
        print(f"wrote the daily results to {out}, with the contracts behind each day's weights")

def main():
    here = Path(__file__).parent
    parser = argparse.ArgumentParser(description="Hold VIX futures long, in the mid term index that "
                                                 "VIXM tracks, and see what that did (no downloading).")
    parser.add_argument("--tenors", type=int_range, nargs=2, metavar=("FIRST", "LAST"),
                        help="contracts held, rolling the first into the last a little each day "
                             "(default: 4 7, the mid term index; 1 2 is the short term index of VXX; "
                             "FIRST equal to LAST holds that one contract, rolled at each settlement). "
                             "Either may be a range, so 1:3 1:3 runs the six positions from 1-1 to 3-3 "
                             "in turn, keeping the pairs whose FIRST is no later than their LAST")
    parser.add_argument("--widths", type=int_range, metavar="N",
                        help="keep only the strips holding this many contracts, as one number or a "
                             "range: 1:3 with --tenors 1:9 1:9 cuts 45 positions to 24, dropping the "
                             "wide strips, which overlap so heavily that they mostly restate each "
                             "other. It prunes what --tenors generates and leaves the contracts named "
                             "by --single-contract alone (default: every width)")
    parser.add_argument("--scale", type=float, default=1.0, metavar="X",
                        help="multiple of the index held, rebalanced daily (default: 1.0)")
    parser.add_argument("--rebalance-days", type=int, default=1, metavar="N",
                        help="bring the position back to the index every N trading days rather than every day, "
                             "since rolling a fraction of a position daily is a fund's job, not an investor's "
                             "(default: 1)")
    parser.add_argument("--single-contract", type=int_range, nargs="?", const="5", metavar="TENOR",
                        help="instead of the index, hold one contract, the TENOR-th out, and roll it on each "
                             "settlement day: one trade a month and no fractions of a position (default TENOR "
                             "if given without one: 5). TENOR may be a range, so 1:6 runs each of the six "
                             "contracts in turn, and it adds to any positions --tenors asks for")
    parser.add_argument("--worst-days", type=int, default=5, metavar="N",
                        help="print the N days the position fell most, with the contracts it held and "
                             "what each of them did; 0 leaves the table out (default: 5)")
    parser.add_argument("--best-days", type=int, default=5, metavar="N",
                        help="the same for the N days it rose most (default: 5)")
    parser.add_argument("--cost", type=float, default=0.05, metavar="C",
                        help="trading cost in VIX points per contract traded (default: 0.05)")
    parser.add_argument("--slices", type=float, nargs="+", default=[0.05, 0.10, 0.20, 0.30],
                        metavar="W",
                        help="fixed slices of SPY notional to hold on top, one row each; SPY on its own "
                             "is always shown to compare with, and a negative slice sells the position "
                             "short instead (default: 0.05 0.1 0.2 0.3)")
    parser.add_argument("--overlay-size", type=float, default=0.15, metavar="W",
                        help="average slice of SPY notional for the sized overlays (default: 0.15)")
    parser.add_argument("--overlay-powers", type=float, nargs="+", default=[0.5, 1.0, 1.5], metavar="P",
                        help="powers of VIX to size the slice by, one row each; about 0.5 holds the hedge "
                             "ratio steady as VIX rises, larger powers bet on crises repeating. The slice "
                             "held constant is always shown to compare with (default: 0.5 1 1.5)")
    parser.add_argument("--shape-power", type=int, default=2, metavar="N",
                        help="highest power of the market's return to fit in the shape regressions, so 3 adds "
                             "cubic terms to the squared ones; each power is fitted both ways and only on the "
                             "downside (default: 2)")
    parser.add_argument("--down-powers", action="store_true",
                        help="also fit the shapes whose powers are switched off above zero, so they bend the "
                             "line only when the market falls; they have not fitted this position as well as "
                             "the shapes that apply across the whole range")
    parser.add_argument("--noreg", action="store_true",
                        help="leave out the regressions of the position's returns on the market's, which say "
                             "what shape the two have but are not needed to follow the strategy")
    parser.add_argument("--min-slice", type=float, default=0.0, metavar="W",
                        help="smallest slice the sized overlays may fall to, so some protection is always on "
                             "(default: 0)")
    parser.add_argument("--max-slice", type=float, default=1.0, metavar="W",
                        help="largest slice they may reach; the unheld rules touch 65%% of the equity notional "
                             "in a crisis (default: 1.0)")
    parser.add_argument("--plot", nargs="?", const="vix_strip.png", metavar="out.png",
                        help="save a chart of the growth of a dollar and the drawdowns")
    parser.add_argument("--output", metavar="daily.csv", help="write the daily results to this file")
    parser.add_argument("--data-dir", default=here / "vix_futures_contracts", metavar="path",
                        help="folder with VX_*_monthly.csv files")
    parser.add_argument("--vix-file", default=here / "vix_spot" / "vix_indexes.csv", metavar="path",
                        help="csv file of CBOE index histories")
    parser.add_argument("--spy-file", default=here / "spy.csv", metavar="path", help="csv file of SPY prices")
    parser.add_argument("--rf-file", default=here / "irx.csv", metavar="path",
                        help="csv of 13 week Treasury bill yields, used to put SPY in excess of the rate")
    parser.add_argument("--vixm-file", default=here / "vixm.csv", metavar="path",
                        help="csv of VIXM's own prices, to check the 4-7 strip against")
    parser.add_argument("--vxx-file", default=here / "vxx.csv", metavar="path",
                        help="csv of VXX's own prices, to check the 1-2 strip against")
    args = parser.parse_args()
    if args.tenors is None and args.single_contract is None:
        args.tenors = [[4], [7]]  # the mid term index, when nothing else is asked for

    pairs = sweep_pairs(args.tenors or [[], []], args.single_contract or [], args.widths)
    if not pairs:
        why = ("no strip of the widths given fits inside the tenors given" if args.widths else
               "--tenors needs a pair whose FIRST is no later than its LAST")
        parser.error(f"no positions to run: {why}")

    # every file the run will write, checked before the work rather than after it
    for asked in (args.output, args.plot):
        if not asked:
            continue
        for path in ([suffixed(asked, *pair) for pair in pairs] if len(pairs) > 1 else [asked]):
            if problem := unwritable(path):
                parser.error(problem)

    df = read_monthly_contracts(args.data_dir)
    contracts = df.sort_values(["Trade Date", "Expiry"]).copy()
    contracts["Price"] = price_series(contracts, "Settle")
    prices = contracts.pivot(index="Trade Date", columns="Expiry", values="Price")
    vix_close = read_vix_indexes(args.vix_file)["VIX"]
    spy_prices = read_prices(args.spy_file)
    if len(pairs) > 1:
        print(f"{len(pairs)} positions to run: "
              f"{', '.join(f'{a}-{b}' for a, b in pairs)}\neach is measured over whatever history its "
              f"own contracts reach back to, which is not the same span for every one of them, so read "
              f"the dates under each heading before reading one against another")
    for number, (first, last) in enumerate(pairs):
        if number:
            print(f"\n{'=' * 110}\n")
        report(first, last, df, prices, vix_close, spy_prices, args, sweeping=len(pairs) > 1)

if __name__ == "__main__":
    main()
