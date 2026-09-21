# VIX futures strategies

Programs for studying VIX futures strategies, together with the CBOE data they read: everything here
runs on the csv files in this repository and downloads nothing. The data, and the tools that gather and
update it, are `vix_utils`, a fork of [dougransom/vix_utils](https://github.com/dougransom/vix_utils),
described under Overview below.

## Holding a strip: `xvix_strip.py`

What holding a strip of VIX futures has done, long or short. A strip is a run of consecutive contracts
held together, with a fraction of the nearest rolled into the furthest every trading day, which is how
the S&P VIX futures indexes are built; the default is the mid term one that VIXM tracks, the contracts
four to seven months out.

```
pip install pandas numpy matplotlib
python xvix_strip.py --plot vix_strip.png
```

[vixm_results.txt](vixm_results.txt) is what that command printed when this was published, so you
can read the whole output, every table below and more, without installing anything. It is written by
the script that builds this release, from the programs and the csv files in it, so it says what the
code here says about the data here.

It reads the csv files here and downloads nothing. `python xvix_strip.py -h` lists the options: another
range of contracts (`--tenors 1 2` is the short term index of VXX), a multiple of the index
(`--scale`, negative to sell the strip short), how often to rebalance, the trading cost assumed, and
where to write the daily results.

Rolling a little of the position every day is what a fund does. Somebody holding a handful of
contracts cannot, so `--rebalance-days 5` rolls weekly instead, and `--single-contract` holds one
contract and rolls it once a month on the settlement day, which is one trade a month and no fractions
of a position. `--tenors 2 2` is the same thing said the other way: one tenor leaves nothing to roll
into, so the position only changes when the front settles and the contract behind it moves up.

Either tenor may be a range, which runs the positions one after another and prints the whole report
for each. `--tenors 1:3 1:3` runs the six from 1-1 to 3-3, `--single-contract 1:6` runs each of the
six contracts held alone, and the two add together, with a position named both ways running once.
`--widths` keeps only the strips holding a given number of contracts, as one number or a range, which
is what keeps a wide sweep short enough to read: with `--tenors 1:9 1:9` it cuts 45 positions to 24,
dropping the wide strips, which overlap so heavily that they mostly restate each other. It prunes what
`--tenors` generates and leaves contracts named by `--single-contract` alone.

A sweep puts the tenors into the names of any files `--plot` and `--output` ask for. Each position is
measured over whatever history its own contracts reach back to, which is not the same span for all of
them, so read the dates under each heading before reading one position against another.

### What it reports

- **What it holds**, with the average maturity it actually held, and how closely it matched the fund
  that holds the same strip, where one does: daily returns correlate 0.94 with VIXM for the 4-7 strip
  and 0.93 with VXX for the 1-2 strip, with differences near those funds' fees, which says the
  construction is right. No fund holds the other strips, so there is nothing to check those against.
- **What it did on its own**: it loses money, at about 16% a year over 2006 to 2026, with a drawdown
  of 98%. Holding volatility long means paying the risk premium that shorting it earns.
- **What the protection costs to carry**, on average and right now, against its own history. Carry is
  what the position earns in a day if the curve does not move, so held long it is the price of the
  insurance; the rest of the return is the curve moving, which is the payout. Over this history a
  dearer carry has not bought a larger payout, so read the current figure as a cost, not a signal.
- **What it did against equities**: a beta to SPY of about -1.2, and more on the days SPY fell (-1.3)
  than on the days it rose (-0.8), which is the shape a hedge is supposed to have.
- **What a slice of it did for a portfolio**: SPY held in full with, say, a tenth of it on top, which
  over this sample cut the worst month from -33% to -28% and the drawdown from -55% to -48% while
  leaving the Sharpe ratio where it was. Futures need no capital, so the slice is added to a full
  holding of SPY rather than funded by selling part of it.
- **What happens when the slice varies with VIX**: SPY's own volatility rises roughly one for one with
  VIX while this position's rises about half as fast, so a fixed slice covers less and less equity
  risk as VIX rises, and holding the hedge ratio steady takes a slice proportional to the square root
  of VIX. Scaling faster than that is a bet on crises repeating, not a risk adjustment.

### What to keep in mind

The case for holding a slice rests on 2008 and 2020. Without those two years it is a steady drag, and
twenty years of data contain only two real tests of it. The numbers are a backtest on daily settlement
prices with an assumed trading cost, they charge nothing for the margin the position ties up, and they
are not advice.

The data are CBOE's, in `vix_futures_contracts` (one file per futures contract) and `vix_spot`; SPY,
the bill yields in `irx.csv` and the fund's prices come from Yahoo Finance through
`xdownload_prices.py`, the only program here that uses the network. Both are described below, with how
to bring them up to date.

## Updating the prices from Yahoo Finance

`spy.csv`, `irx.csv` (13 week Treasury bill yields) and the funds' own prices in `vixm.csv` and
`vxx.csv` are snapshots taken when
this was published. To refresh them:

```
pip install yfinance
python xupdate_data.py
```

That runs all of them: the CBOE futures and index histories through `vixutil`, and the Yahoo Finance
series below through `xdownload_prices.py`. It prints where each file ended before and after, and says
so if they do not all end on the same date, which matters because the tables are joined on the dates
every file has: a stale spot file silently shortens them, and a stale price file is carried forward and
reads as a day the market did not move. `--dry-run` shows where they stand without downloading.

The individual commands, if you want one of them:

```
python xdownload_prices.py SPY
python xdownload_prices.py ^IRX irx.csv
python xdownload_prices.py VIXM
python xdownload_prices.py VXX
```

Use the `Adj Close` column, as the programs here do: VIXM had a reverse split in 2021, so its
unadjusted closes are not comparable across it.

## Data directory `vix_futures_contracts`

The `vix_futures_contracts` directory holds the output of `vixutil --contracts vix_futures_contracts`, downloaded on 2026-09-18:
668 contracts (273 monthly and 395 weekly expiries) with 51,114 daily records from 2004-03-26 to 2026-09-17.
Contracts that had not expired by then contain data only up to that date; their `Expired` column is `False`.
To update the files, rerun the command. Previously downloaded CBOE files are cached, so only new data is fetched.

Each file has these columns:

- `Trade Date`, `Expiry`, `Futures` (CBOE contract name)
- `Open`, `High`, `Low`, `Close`, `Settle`, `Change`, `Total Volume`, `EFP`, `Open Interest`
- `Weekly`: whether the contract has a weekly (rather than monthly) expiry
- `Tenor_Monthly`, `Tenor_Days`, `Tenor_Trade_Days`: time to expiry in monthly expiries, calendar days, and trading days
- `Year`, `MonthOfYear` (of the expiry), `File` (source CBOE file), `Expired`

Prices before 2007-03-26 were quoted by CBOE at 10 times the VIX level; they have been divided by 10 so the whole history is on one scale.
Where CBOE's archive (2004-2013) and current files overlap, duplicate rows are dropped, as are rows with a closing price of 0,
except on a contract's settlement day (see the fixes above).

## Data directory `vix_spot`

`vix_spot/vix_indexes.csv` holds CBOE daily index histories, written by `vixutil -c vix_spot/vix_indexes.csv` on 2026-09-18,
in record format with columns `Trade Date`, `Symbol`, `Open`, `High`, `Low`, `Close` (the first, unnamed column is a row number).
Spot VIX (`VIX`) runs from 1990-01-02; the other symbols are VIX1D, VIX9D, VIX3M, VIX6M (open/high/low/close) and
VVIX, GVZ, OVX, SHORTVOL, LONGVOL, VXTLT (close only), each starting when CBOE began publishing it.
VIX closes are taken at 4:15 pm ET, which is later than the VIX futures settlement prices are determined.
`read_vix_indexes()` in `xread_data.py` loads it as a table of closes with one column per symbol.

## Overview

*vix_utils* provides some tools for preparing data for analysing  the VIX Futures and Cash Term structures.

The futures can also contain a 30 day continuous maturity weighting of front two months of vix futures.

VIX Futures Data downloaded from [CBOE Futures Historical Data](https://www.cboe.com/us/futures/market_statistics/historical_data/).

Vix Cash Data are downloaded from [CBOE Historical Volatility Indexes](https://www.cboe.com/tradable_products/vix/vix_historical_data/).

There is an API for Python to load the data into Pandas DataFrames.  If you do your analysis in Python, use the API.

Since there is no documentation yet, look at the examples in the src/vix_utils/examples folder.
There is a Jupyter Notebook vix_utils.ipynb in that folder.

*Important note for Juypter notebooks.*  
You must use  async_get_vix_index_histories and async_load_vix_term_structure 
rather than get_vix_index_histories and load_vix_term_structure.  There is an example Jupyter notebook "vix_utils use in Jupyter.ipynb" in the src/vix_utils/examples folder. 
 
If you do your analysis in other tools such as R or excel, you can use the command line tool vixutil.

`vixutil -h` will give the help.  The data are availble in record and wide formats.  Just run it and look at the excel or csv output to see what they look like.

## Installation

You will need a Python 3.11 or later instalation.

### Install from the Python Packaging Index
 
Install using pip from [The Python Package Index ](https://www.pypi.org):

`pip install vix_utils`

If you want to run the samples, install like this:
`pip install vix_utils[examples]`

The sample to load all the various data frames can be run as:
'vix_sample_load_data'

The sample to plot the history of futures and cash term structures:
`vix_sample_plots`

To load the sample Jupyter notebook, run vix_sample_load_data to figure out where the examples folder is. Browse there with Jupyter and open a notebook.   

### Development 

Clone from  [github repository](https://github.com/vivek-v-rao/vix_futures_strategies).

 
`pip install -e .[test,examples]` will:
- install vix_utils into your python environment, including any command line scripts. 
- install the necessary prequisites for running any 
tests in the `test` folder, and for running the programs in the `src/vixutils/examples` folder.
