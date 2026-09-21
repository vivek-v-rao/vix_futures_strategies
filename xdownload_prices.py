"""
Download daily prices of a symbol from Yahoo Finance (requires yfinance) and save them to a csv file
with columns Date, Open, High, Low, Close, Adj Close (adjusted for dividends and splits), Volume.

usage: python xdownload_prices.py SYMBOL [output.csv] [start_date]
       the output file defaults to the lower case symbol with a .csv suffix, such as spy.csv
"""
import sys

import pandas as pd
import yfinance as yf

def download_prices(symbol="SPY", path=None, start="1900-01-01") -> pd.DataFrame:
    """Download the daily price history of symbol and write it to path, which defaults to the lower
    case symbol with a .csv suffix. Returns the prices."""
    d = yf.download(symbol, start=start, auto_adjust=False, progress=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d.index.name = "Date"
    d = d[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    d.to_csv(path or f"{symbol.lower()}.csv", float_format="%.6f")
    return d

def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print(__doc__.strip())
        return
    symbol = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    path = sys.argv[2] if len(sys.argv) > 2 else f"{symbol.lower()}.csv"
    start = sys.argv[3] if len(sys.argv) > 3 else "1900-01-01"
    d = download_prices(symbol, path, start)
    print(f"wrote {len(d)} days of {symbol} from {d.index.min().date()} to {d.index.max().date()} to {path}")

if __name__ == "__main__":
    main()
