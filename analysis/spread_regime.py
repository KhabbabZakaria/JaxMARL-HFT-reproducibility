"""Measure the quoted-spread regime of each ticker in the raw LOBSTER book files.

This is the evidence behind the ticker-regime finding, and it has two halves.
The market-making agent quotes a fixed number of ticks away from mid
(spread_multiplier=3.0 with multiplier_type='tick'), so whether that lands inside
or behind the touch is decided by how wide the book is in ticks. And when there is
no room inside the spread, the only option left is to join the touch queue, where
the agent's one-share order (fixed_quant_value=1) ranks behind whatever is already
resting there. Both are properties of the ticker, not of the policy, so this script
measures both: the spread in ticks and the depth at the touch.

Needs the raw LOBSTER CSVs, which are licensed and not in this repo. The output
it produces (results/spread_regime.csv) is committed, so the finding can be read
without them.

Usage:  python analysis/spread_regime.py [--data data/rawLOBSTER] [--rows 400000]
"""
import argparse
import glob
import pathlib

import numpy as np
import pandas as pd

TICK = 100      # LOBSTER prices are in 1/10000 dollars; tick_size=100 => $0.01
DOLLARS = 10000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/rawLOBSTER")
    ap.add_argument("--rows", type=int, default=400_000,
                    help="rows per file; the book has millions and the regime is stable")
    ap.add_argument("--out", default="results/spread_regime.csv")
    args = ap.parse_args()

    files = sorted(glob.glob(f"{args.data}/*/*/*orderbook_10.csv"))
    if not files:
        raise SystemExit(f"no orderbook CSVs under {args.data} "
                         "(LOBSTER data is licensed and not shipped with this repo)")

    rows = []
    for f in files:
        parts = pathlib.Path(f).parts
        ticker, period = parts[-3], parts[-2]
        # Columns 0-3 of a LOBSTER orderbook file are ask price, ask size, bid
        # price, bid size at level 1.
        d = pd.read_csv(f, header=None, usecols=[0, 1, 2, 3],
                        names=["ask", "ask_sz", "bid", "bid_sz"], nrows=args.rows)
        d = d[(d.ask > 0) & (d.bid > 0) & (d.ask < 1e9)]
        spread = (d.ask - d.bid) / TICK
        mid = (d.ask + d.bid) / 2 / DOLLARS
        # Both sides pooled: the agent quotes two-sided, so it queues against either.
        touch_depth = pd.concat([d.ask_sz, d.bid_sz])
        rows.append({
            "ticker": ticker, "period": period,
            "median_mid_usd": round(float(mid.median()), 2),
            "median_spread_ticks": float(spread.median()),
            "mean_spread_ticks": round(float(spread.mean()), 2),
            "pct_spread_at_one_tick": round(float(np.mean(spread <= 1) * 100), 1),
            "median_touch_depth_shares": int(touch_depth.median()),
            "mean_touch_depth_shares": int(touch_depth.mean()),
            "p90_touch_depth_shares": int(touch_depth.quantile(0.9)),
            "n_rows": int(len(d)),
        })
        print(f"{ticker:5s} mid=${rows[-1]['median_mid_usd']:8.2f} "
              f"median spread={rows[-1]['median_spread_ticks']:5.1f} ticks  "
              f"at-1-tick={rows[-1]['pct_spread_at_one_tick']:5.1f}%  "
              f"median touch depth={rows[-1]['median_touch_depth_shares']:7d} shares")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
