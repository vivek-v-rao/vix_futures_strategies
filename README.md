# VIX Utils

This project is a fork of [dougransom/vix_utils](https://github.com/dougransom/vix_utils).

## Additions in this fork

- **One CSV file per futures contract.** `vixutil --contracts <output_dir>` writes the daily history of every VIX futures
  contract, weekly and monthly expiries, to its own file named `VX_<expiry>_monthly.csv` or `VX_<expiry>_weekly.csv`.
  The same thing is available from Python as `vix_utils.write_futures_contracts_csv(records, out_dir)`, where `records`
  is the DataFrame returned by `load_vix_term_structure()`. Rows are sorted by trade date and prices are rounded to 4 decimals.
- **Faster `vixutil` runs.** When only `--contracts` is requested, `vixutil` no longer downloads the VIX cash index histories
  or builds the other term structure outputs.
- **Scripts for analysis and trading research**, described below: `xcontract_stats.py` (statistics and regressions),
  `xtrade_carry.py` (backtested trading strategies), `xwrite_curve.py` (the whole curve as one csv) and
  `xdownload_prices.py` (Yahoo Finance prices). They read only the csv files in this repository and download nothing.
- **Fixes to the underlying package:**
  - Prices on 2007-03-26 were being divided by 10 a second time. CBOE's files are already on the new scale that day,
    so only prices *before* it are rescaled now.
  - Contracts whose data stops before expiry (July and October 2004) are treated as expired.
  - The settlement-day row of contracts expiring before 2015 is kept. CBOE recorded open, high, low and close as 0 that
    day and filled in only the settlement price, so those rows were being dropped and each contract lost its final value.
    The settlement price is now used as that day's close, with the other prices left empty, for contracts that had open
    interest or volume. This restored 123 rows.
  - Three pandas deprecation warnings.

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

## Analysis scripts

These four scripts sit at the top of the repository and need only pandas, with statsmodels for the regressions and
yfinance for the downloader. Each takes `-h` for help.

### `xcontract_stats.py` — statistics, regressions and fitted models

Reads the monthly contract files into one DataFrame and reports on them. Nothing is downloaded. Run
with no arguments it prints the ten tables that describe the contracts.

**[MODELS.md](MODELS.md) gives the formula behind every model below, how each is estimated, and what
the quantities mean.** It deliberately carries no fitted numbers: those depend on the sample, and the
sample matters here — the correlation across the curve is materially flatter since 2018 than before.
Use `--min-date` and `--max-date` to fit a window and read the numbers off your own run.

- `--summary` — one row per contract: days of data, first and last trade dates, and first, last, high and low closes.
- `--return-stats` — daily return statistics by trading days until a contract's last close: count, median, mean, standard
  deviation, annualized Sharpe ratio, skew, excess kurtosis, minimum and maximum. `--buckets 1 20` prints one table per
  bucket width, `--max-days` limits the horizon, `--log-returns` switches to log returns.
- `--carry-regression` — regresses each day's return on the previous day's carry per day, for the front contract against
  spot VIX and for later contracts against the next one closer to expiration. Standard errors are clustered by trade
  date, with Driscoll-Kraay (`--hac-lags`) as a check; `--winsorize` clips the return tails.
- `--term-regressions` — regresses each contract's price on the price of the contract closer to expiration (the front
  contract on spot VIX), in levels, logs or daily changes, with the square root of days to expiry as an optional term.
  The `t_Slope_1` column tests whether the slope is 1.

Volatility, fitted:

- `--vol-decay` — how the spread of a day's return grows as expiration approaches, as a power law, an
  exponential and a shifted power, with the fall in carry fitted in the same shapes beside it.
- `--cev` — the size of a move against the level it moved from and the days it has left, as a power
  of the level and as a displacement from a floor, fitted pooled and within each day; `--rising-floor`
  lets the floor grow with maturity.
- `--riskmetrics` — whether the size of a move still clusters in time once level and maturity are
  accounted for; `--rm-lambda` sweeps the memory and `--rm-split` divides fitting from scoring.
- `--asymmetry` — whether an up move leaves the next day livelier than a down move of the same size.
- `--innovations` — which distribution the moves came from once divided by the size expected of them:
  normal, Student *t*, or Hansen's skewed *t*. `--skew-by-level` cuts the tilt by price.
- `--slope` — whether a flatter curve means livelier futures, over `--slope-steps` of the curve.
- `--vvix` — whether the implied volatility of VIX says anything the price of the future does not.

Correlation:

- `--correlations` — the plain correlation matrix of daily returns: SPY, spot VIX, and the curve.
- `--corr-decay` — that matrix as a model, correlation declining in the trading days between two
  expirations, which a pair keeps as it rolls, with the maturity effect fitted five ways.
- `--corr-ewma` — a decayed correlation scored against those shapes and against two ways of mixing
  the two, on days none of them were fitted on; `--corr-decays` sets the memories tried.
- `--pair-matrix` — the realized correlations of named contracts over the days they actually shared,
  with the days behind every cell beside them. `--pair-contracts VXF27 VXG27` names them.

Every option takes an optional file name to also write the table as csv. `--min-date` and `--max-date`
narrow the window before anything is computed, `--terse` names each table without reading it, and
`--notime` suppresses the timing table each run ends with.

### `xtrade_carry.py` — backtests

**[STRATEGIES.md](STRATEGIES.md) describes each strategy, gives the command that runs it, and compares their historical
performance on one footing, with the caveats that go with backtests.**

The strategies below share one set of machinery: trade lag, volatility targeting, no-trade bands, trading costs, an optional SPY
hedge (`--hedge-window`, `--hedge-scale`), performance statistics, results by year, and beta to SPY. `--backtest out.csv`
writes the daily results.

- `--strategy carry` (default) — expected return of each contract is a walk-forward regression slope times its carry,
  sized by risk over the tenors in `--tenors`, with a regime filter for an inverted curve.
- `--strategy basis` — the rule of Simon and Campasano (2014): trade the nearest contract when its basis per day passes
  `--threshold`, hedged with SPY.
- `--strategy benchmark` — a constant maturity position with no signal (`--target-days`, `--side`), interpolated between
  the two contracts that bracket the target. The yardstick for the rest.
- `--strategy index` — an S&P style index that holds `--index-tenors FIRST LAST` and rolls the first into the last a
  little each day, the construction the volatility ETPs track: `1 2` is the short term index of VXX and `4 7` the mid
  term index of VIXM. `--scale` takes the signed multiple held, rebalanced daily as the products are, so 1.5 is UVXY,
  −0.5 today's SVXY and −1 the XIV note. Daily returns correlate 0.93 with VXX, 0.94 with VIXM, 0.99 with UVXY and 0.98
  with SVXY.
- `--strategy slope` — a level neutral calendar spread on the term structure slope, after Johnson (2017), either timed by
  the slope score or held constantly (`--slope-timing`).
- `--strategy vrp` — short volatility when VIX is above realized S&P volatility over `--rv-window` days, long when below.
- `--strategy residual` — sell contracts that are rich against a curve fitted to each day's prices and buy the cheap ones.

### `xwrite_curve.py` — the curve as one csv

Writes one row per trade date with the CBOE volatility indexes, SPY, and by tenor the futures prices, calendar and
trading days to expiry, contract symbols (VXU26) and roll adjusted returns, with carry under `--carry`. Intended for use
outside Python, and for seeing the curve and its rolls in one place.

### `xdownload_prices.py` — Yahoo Finance prices

`python xdownload_prices.py SPY` writes `spy.csv`, the only script here that reaches the network. The SPY history in this
repository was written by it and is used for the beta calculations, the hedges and the variance risk premium.

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

Clone from  [github repository](https://github.com/vivek-v-rao/vix_utils).

 
`pip install -e .[test,examples]` will:
- install vix_utils into your python environment, including any command line scripts. 
- install the necessary prequisites for running any 
tests in the `test` folder, and for running the programs in the `src/vixutils/examples` folder.

## Examples
Source is in `src/vix_utils/examples`
 
~~~
## Data Notes
These dates appear to be missing from the CBOE Data.
At some point they need to be patched in if they exist.
```
[Timestamp('2006-11-10 00:00:00'), Timestamp('2007-01-03 00:00:00'), Timestamp('2021-04-02 00:00:00'), Timestamp('2021-12-24 00:00:00')]
```
There seem to be  a few dates where spot indexes are missing, you will have to workaround by using fill feature of Pandas datafame, or skip those days, in any analysis.
~~~

