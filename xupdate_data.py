"""
Bring every price file this project reads up to date, in one command.

The data arrive from two places and four commands, and running only some of them leaves the files at
different dates. That is quiet rather than loud: a futures file a day ahead of the spot index file
silently shortens every table, because the backtest is joined to VIX and SPY and the join drops what
is missing, and a stale SPY file is worse, since its last price is carried forward and reads as a day
the market did not move.

So this runs them all and prints where each file ended before and after, and says plainly if they do
not agree at the end. It downloads; nothing else in this project does except xdownload_prices.py.

vixutil comes from the vix_utils package in src, and only re-downloads contracts that are still
listed: a settled contract is cached and skipped, so a refresh is quick. The Yahoo Finance steps need
yfinance, which the others do not, and are reported as skipped rather than fatal if it is missing.

usage: python xupdate_data.py [--only NAME ...] [--dry-run] [--data-dir path]
"""
import argparse
import subprocess
import sys
from pathlib import Path

# name, the command to run, the file or folder it writes, and which column of that file holds the date
SOURCES = [
    ("futures", ["-m", "vix_utils.vixutil", "--contracts", "vix_futures_contracts"],
     "vix_futures_contracts", 0),
    ("spot", ["-m", "vix_utils.vixutil", "-c", "vix_spot/vix_indexes.csv"],
     "vix_spot/vix_indexes.csv", 1),
    ("spy", ["xdownload_prices.py", "SPY"], "spy.csv", 0),
    ("rate", ["xdownload_prices.py", "^IRX", "irx.csv"], "irx.csv", 0),
    ("vixm", ["xdownload_prices.py", "VIXM"], "vixm.csv", 0),
    ("vxx", ["xdownload_prices.py", "VXX"], "vxx.csv", 0),
]

def last_date(target: Path, column: int) -> str:
    """The latest date in a csv file, or across the monthly contract files of a folder.

    Read rather than parsed: a date written the way these files write it sorts as text, and this needs
    no pandas, so the updater runs even where the rest does not.
    """
    files = sorted(target.glob("*_monthly.csv")) if target.is_dir() else [target]
    latest = ""
    for path in files:
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as handle:
            next(handle, None)  # the heading
            for line in handle:
                fields = line.split(",")
                if len(fields) > column:
                    value = fields[column].strip()
                    if len(value) == 10 and value[4] == "-" and value > latest:
                        latest = value
    return latest or "nothing"

def main():
    parser = argparse.ArgumentParser(description="Bring every price file up to date (this downloads).")
    parser.add_argument("--only", nargs="+", metavar="NAME",
                        choices=[name for name, _, _, _ in SOURCES],
                        help=f"update only these: {', '.join(name for name, _, _, _ in SOURCES)} "
                             f"(default: all of them)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the commands and where the files end now, and download nothing")
    args = parser.parse_args()

    here = Path(__file__).parent
    wanted = [s for s in SOURCES if not args.only or s[0] in args.only]
    failures, rows = [], []
    for name, command, target, column in wanted:
        path = here / target
        before = last_date(path, column)
        shown = " ".join(["python"] + command)
        if args.dry_run:
            rows.append((name, target, before, "would run", shown))
            continue
        print(f"{name}: {shown}")
        done = subprocess.run([sys.executable] + command, cwd=here, capture_output=True, text=True)
        after = last_date(path, column)
        if done.returncode:
            last_line = (done.stderr.strip().splitlines() or ["no message"])[-1]
            failures.append(f"{name}: {last_line[:150]}")
            rows.append((name, target, before, after, "FAILED"))
        else:
            rows.append((name, target, before, after, "" if after != before else "unchanged"))

    width = max(len(target) for _, _, target, _ in wanted)
    print()
    for name, target, before, after, note in rows:
        print(f"{name:<8} {target:<{width}}  {before} -> {after}  {note}")

    # after a run it is where the files ended up; on a dry run, where they stand now
    ends = {before if args.dry_run else after
            for _, _, before, after, note in rows if note != "FAILED"}
    if len(ends) > 1:
        print(f"\nthe files do not end on the same date: {', '.join(sorted(ends))}; the tables are "
              f"joined on the dates every file has, so the earliest of these is where they stop")
    elif ends:
        print(f"\nevery file now ends {ends.pop()}")
    if failures:
        print("\nfailed:")
        for failure in failures:
            print(f"   {failure}")
        print('if yfinance is missing, "pip install yfinance" fixes the Yahoo Finance steps')
    return 1 if failures else 0

if __name__ == "__main__":
    sys.exit(main())
