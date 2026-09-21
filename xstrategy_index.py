"""
The S&P style VIX futures indexes: hold the contracts from one tenor to another and roll a fraction of
the nearest into the furthest on every trading day of the roll period. Contracts 1 to 2 is the short
term index that VXX tracks and 4 to 7 the mid term index of VIXM.

Kept in its own module so that a program can use it without the rest of the backtests.
"""
import numpy as np
import pandas as pd

from xread_data import add_returns, price_series, trading_calendar

def first_unusable_date(df: pd.DataFrame, max_tenor: int):
    """First trade date on which a contract of tenor <= max_tenor cannot be dated, or None.

    A contract is usable once the trading days to its expiry can be counted. That used to mean waiting
    for it to settle, which cut the end off every backtest, one expiry for each tenor held, so a strip
    of the fourth to seventh contracts stopped seven months before the data did. An expiry is fixed by
    exchange rule and published years ahead, so the count comes from the calendar and a contract still
    trading is as usable as a settled one. This now finds a date only where the calendar runs out.

    Tenor 1 is the contract nearest to expiration on each trade date.
    """
    a = df.sort_values(["Trade Date", "Expiry"])
    tenor = a.groupby("Trade Date").cumcount() + 1
    undated = ~a["Expiry"].isin(trading_calendar(a["Trade Date"]))
    dates = a.loc[(tenor <= max_tenor) & undated, "Trade Date"]
    return dates.min() if len(dates) else None

def sp_roll_fraction(c: pd.DataFrame) -> pd.Series:
    """The share of the roll still to do on each trade date, from the S&P VIX futures index rules.

    The indexes roll a fixed fraction of the nearest contract into a later one on each trading day of
    the roll period, which runs from one monthly settlement to the next. The fraction left is the
    trading days the nearest holdable contract has to its last close, over the trading days of its
    whole roll period: 1 on the day a contract becomes the nearest one, falling to 1/dt the day before
    it settles. c is the frame from add_returns with a Tenor column.
    """
    front = c[c["Tenor"] == 1].set_index("Trade Date")
    remaining = front["Days_To_Last_Close"]
    # the roll period of a contract is as long as its first day as the nearest contract
    period = remaining.groupby(front["Expiry"]).transform("max")
    return (remaining / period).where(period > 0)

def index_weights(fraction: float, first_tenor: int, last_tenor: int) -> dict:
    """Weights by tenor of an S&P VIX futures index that holds first_tenor..last_tenor.

    With n = last_tenor - first_tenor contracts' worth of position, the nearest tenor holds
    fraction / n, each tenor in between holds 1 / n, and the last holds (1 - fraction) / n, so the
    position rolls from the nearest to the furthest over the roll period and the weights sum to 1.
    (1, 2) is the short-term index that VXX tracks and (4, 7) the mid-term index of VIXM.
    """
    n = last_tenor - first_tenor
    weights = {tenor: 1 / n for tenor in range(first_tenor + 1, last_tenor)}
    weights[first_tenor] = fraction / n
    weights[last_tenor] = (1 - fraction) / n
    return weights

def backtest_index(df: pd.DataFrame, first_tenor: int = 1, last_tenor: int = 2, side: int = 1,
                   weight: float = 1.0, cost: float = 0.05, price: str = "Settle",
                   trading_days_per_year: float = 252, rebalance_days: int = 1) -> pd.DataFrame:
    """Backtest an S&P style VIX futures index, the construction the volatility ETPs track.

    Each day the contracts from first_tenor to last_tenor are held in the weights of index_weights, so
    a fraction of the nearest contract rolls into the furthest on every trading day of the roll period.
    (1, 2) reproduces the S&P 500 VIX Short-Term Futures index that VXX tracks and (4, 7) the Mid-Term
    index of VIXM; side 1 is long the index and -1 short it. The indexes are calculated from daily
    settlement prices, so price defaults to Settle rather than Close. Costs are |change in weight| * cost / price
    with cost in VIX points. Unlike backtest_constant_maturity, which interpolates to an exact maturity,
    the average maturity here rises and falls through the roll, as the real indexes do.
    With rebalance_days above 1 the weights are only brought back to the index every that many trading
    days, and held in between. Rolling a little every day is what a fund does; an investor holding a
    handful of contracts cannot move a twentieth of a position daily, and this says what the coarser
    schedule costs. It barely changes how much is traded over a month, since the same notional has to
    move either way; what changes is the number of trades and how far the position drifts from the
    index between them.

    The result has one row per trade date: Gross_Return and Net_Return, the cost and turnover of the
    day's trades, the weights held by tenor, the average maturity of the position and the share of the
    roll still to do.
    """
    # days to last close are worked out before the data is cut, so the last day is not mistaken for
    # a contract's expiry
    end = first_unusable_date(df, last_tenor)
    c = add_returns(df).sort_values(["Trade Date", "Expiry"])
    if end is not None:
        c = c[c["Trade Date"] < end]
    # a contract at its last close cannot be held further, so tenor 1 is the nearest one that can be
    c["Tenor"] = c[c["Days_To_Last_Close"] > 0].groupby("Trade Date").cumcount() + 1
    c = c.sort_values(["Expiry", "Trade Date"])
    c["Price"] = price_series(c, price)  # the close stands in where the settlement cannot be believed
    c["PnL_Return"] = c["Price"] / c.groupby("Expiry")["Price"].shift(1) - 1
    dates = pd.Index(sorted(c["Trade Date"].unique()))
    c["i"] = dates.get_indexer(c["Trade Date"])

    # every contract is kept, including on its last day, so its final return is earned before it goes
    rows = {}
    for r in c.itertuples(index=False):
        rows.setdefault(r.i, {})[r.Expiry] = (r.Price, r.Tenor, r.Tenor_Days, r.PnL_Return,
                                              r.Days_To_Last_Close)
    fractions = sp_roll_fraction(c).reindex(dates).to_numpy()

    held, records = {}, []
    for i in range(len(dates)):
        today = rows.get(i, {})
        gross = sum(w * today[e][3] for e, w in held.items() if e in today and not np.isnan(today[e][3]))

        record = {"Trade Date": dates[i], "Gross_Return": gross,
                  "Gross_Leverage": sum(abs(w) for w in held.values()), "Net_Exposure": sum(held.values()),
                  "Roll_Fraction": fractions[i]}
        for tenor in range(1, last_tenor + 1):
            record[f"Weight_{tenor}"] = 0.0
        record["Weight_Other"] = 0.0
        yesterday = rows.get(i - 1, {})
        maturity = 0.0
        for e, w in held.items():
            tenor = yesterday[e][1] if e in yesterday else np.nan
            slot = f"Weight_{int(tenor)}" if np.isfinite(tenor) and tenor <= last_tenor else "Weight_Other"
            record[slot] += w
            if e in yesterday:
                maturity += abs(w) * yesterday[e][2]
        total = sum(abs(w) for w in held.values())
        record["Maturity_Days"] = maturity / total if total else np.nan

        by_tenor = {tenor: e for e, (_, tenor, _, _, days_left) in today.items() if days_left > 0}
        targets = dict(held)  # between rebalances the position stands, whatever the index has done
        settling = any(e in today and today[e][4] == 0 for e in held)  # a contract settling must be replaced
        rebalancing = i % rebalance_days == 0 or settling or not held
        if rebalancing:
            targets = {}
        if rebalancing and np.isfinite(fractions[i]) and all(t in by_tenor
                                                             for t in range(first_tenor, last_tenor + 1)):
            for tenor, w in index_weights(fractions[i], first_tenor, last_tenor).items():
                targets[by_tenor[tenor]] = side * weight * w

        cost_return = turnover = 0.0
        for e in set(held) | set(targets):
            target, current = targets.get(e, 0.0), held.get(e, 0.0)
            if target != current and e in today:
                turnover += abs(target - current)
                cost_return += abs(target - current) * cost / today[e][0]
                held[e] = target
        # a contract at its last close is settled, after earning the day's return above
        held = {e: w for e, w in held.items() if w != 0 and not (e in today and today[e][4] == 0)}

        record.update({"Cost": cost_return, "Net_Return": gross - cost_return, "Turnover": turnover,
                       "Inverted": False})
        record["Positions"] = dict(held)
        records.append(record)
    return pd.DataFrame(records).set_index("Trade Date")

def backtest_single_contract(df: pd.DataFrame, tenor: int = 5, side: int = 1, weight: float = 1.0,
                             cost: float = 0.05, price: str = "Settle") -> pd.DataFrame:
    """Hold one contract and roll it once a month, on the day the front contract settles.

    On each settlement day the position moves into the contract that is then tenor places out, and it
    is held untouched until the next settlement. This is what somebody holding a contract or two can
    actually do: one trade a month, no fractions of a position.

    The maturity of what is held therefore falls through the month, from about tenor months to about
    tenor minus one, so the average sits near the middle: holding the fifth contract averages about
    four and a half months, not five. The result has the columns of backtest_index.
    """
    end = first_unusable_date(df, tenor)
    c = add_returns(df).sort_values(["Trade Date", "Expiry"])
    if end is not None:
        c = c[c["Trade Date"] < end]
    c["Tenor"] = c[c["Days_To_Last_Close"] > 0].groupby("Trade Date").cumcount() + 1
    c = c.sort_values(["Expiry", "Trade Date"])
    c["Price"] = price_series(c, price)
    c["PnL_Return"] = c["Price"] / c.groupby("Expiry")["Price"].shift(1) - 1
    dates = pd.Index(sorted(c["Trade Date"].unique()))
    c["i"] = dates.get_indexer(c["Trade Date"])

    rows = {}
    for r in c.itertuples(index=False):
        rows.setdefault(r.i, {})[r.Expiry] = (r.Price, r.Tenor, r.Tenor_Days, r.PnL_Return,
                                              r.Days_To_Last_Close)
    settlements = {i for i, day in rows.items() if any(v[4] == 0 for v in day.values())}

    held_expiry, held_weight, records = None, 0.0, []
    for i in range(len(dates)):
        today = rows.get(i, {})
        contract = today.get(held_expiry)
        gross = held_weight * (contract[3] if contract and np.isfinite(contract[3]) else 0.0)

        record = {"Trade Date": dates[i], "Gross_Return": gross, "Gross_Leverage": abs(held_weight),
                  "Net_Exposure": held_weight, "Roll_Fraction": np.nan}
        yesterday = rows.get(i - 1, {})
        for slot in range(1, tenor + 1):
            record[f"Weight_{slot}"] = 0.0
        record["Weight_Other"] = 0.0
        if held_weight and held_expiry in yesterday:
            slot = yesterday[held_expiry][1]
            record[f"Weight_{int(slot)}" if np.isfinite(slot) and slot <= tenor else "Weight_Other"] += held_weight
            record["Maturity_Days"] = yesterday[held_expiry][2]
        else:
            record["Maturity_Days"] = np.nan

        cost_return = turnover = 0.0
        if i in settlements or held_expiry is None:
            wanted = {t: e for e, (_, t, _, _, days_left) in today.items() if days_left > 0}.get(tenor)
            if wanted is not None:
                if held_weight and held_expiry in today:  # out of the old one
                    turnover += abs(held_weight)
                    cost_return += abs(held_weight) * cost / today[held_expiry][0]
                held_expiry, held_weight = wanted, side * weight
                turnover += abs(held_weight)  # and into the new one
                cost_return += abs(held_weight) * cost / today[wanted][0]
        if held_expiry is not None and held_expiry in today and today[held_expiry][4] == 0:
            held_expiry, held_weight = None, 0.0

        record.update({"Cost": cost_return, "Net_Return": gross - cost_return, "Turnover": turnover,
                       "Inverted": False})
        record["Positions"] = {held_expiry: held_weight} if held_weight else {}
        records.append(record)
    return pd.DataFrame(records).set_index("Trade Date")
