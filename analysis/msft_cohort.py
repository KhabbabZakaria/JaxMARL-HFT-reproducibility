"""Large-tick seed cohort: three seeds of MSFT alone, against the small-tick cohort.

The large-tick finding originally rested on one run, which is the error this study
documents elsewhere. This compares a three-seed MSFT cohort with the four-seed
AAPL/AMZN/GOOG cohort on the metrics that carry the claim:

  - whether the market maker ever quotes at the touch (action 0)
  - whether it accumulates any inventory at all
  - whether the execution agent is affected (it should not be)

Reads the in-progress logs at the repo root, so it can be run before the seeds
finish; it truncates every run to the shortest one so the comparison stays fair.

Usage:  python analysis/msft_cohort.py [--budget N]
"""
import argparse
import pathlib
import statistics
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from extract_metrics import parse_log, COLUMNS, EPISODE_STEPS, TASK_SIZE  # noqa: E402
from mm_action_choice import BID_OFFSETS, ASK_OFFSETS, POSTS  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
SMALL_TICK = {2: "run_AAPL_AMZN_GOOG", 7: "run_AAPL_AMZN_GOOG_seed7",
              13: "run_AAPL_AMZN_GOOG_seed13", 42: "run_AAPL_AMZN_GOOG_seed42"}


def at_touch_pct(df):
    """Share of updates whose most-used action posts on both sides at the touch."""
    ok = {a for a in range(len(POSTS))
          if POSTS[a] and BID_OFFSETS[a] == 0 and ASK_OFFSETS[a] == 0}
    return df["mm_top_action"].isin(ok).mean() * 100


def describe(df, label, seed):
    tail = df.tail(max(1, len(df) // 10))
    doom = tail["exe_doom"].mean()
    return {
        "cohort": label, "seed": seed, "updates": len(df),
        "mm_reward": tail["reward_mm"].mean(),
        "mm_pnl": tail["mm_pnl"].mean(),
        "mm_peak_abs_inventory": df["mm_inv"].abs().max(),
        "mm_at_touch_pct": at_touch_pct(tail),
        "exe_reward": tail["reward_exe"].mean(),
        "exe_fill_rate_pct": (1 - doom * EPISODE_STEPS / TASK_SIZE) * 100,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=0,
                    help="updates to truncate every run to; 0 = shortest MSFT run")
    args = ap.parse_args()

    msft = sorted(ROOT.glob("run_MSFT_seed*.log"))
    if not msft:
        raise SystemExit("no run_MSFT_seed*.log at the repo root")

    parsed = {int(p.stem.split("seed")[1]): pd.DataFrame(parse_log(p), columns=COLUMNS)
              for p in msft}
    budget = args.budget or min(len(d) for d in parsed.values())
    if budget < 1:
        raise SystemExit("MSFT runs have not produced a complete update yet")
    print(f"comparing at {budget} updates "
          f"({'in progress' if budget < 975 else 'full study budget'})\n")

    rows = [describe(d.head(budget), "large-tick (MSFT)", s) for s, d in sorted(parsed.items())]
    for seed, run in sorted(SMALL_TICK.items()):
        d = pd.read_csv(ROOT / "results" / "metrics" / f"{run}.csv")
        rows.append(describe(d.head(budget), "small-tick (AAPL,AMZN,GOOG)", seed))

    out = pd.DataFrame(rows)
    print(out.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

    print("\ncohort means:")
    for label, g in out.groupby("cohort", sort=False):
        print(f"  {label:28s} at-touch {g.mm_at_touch_pct.mean():5.1f}%  "
              f"peak|inv| {g.mm_peak_abs_inventory.mean():6.3f}  "
              f"PnL {g.mm_pnl.mean():10,.0f}  "
              f"EXE fill {g.exe_fill_rate_pct.mean():5.1f}%")

    msft_rows = out[out.cohort.str.startswith("large")]
    if len(msft_rows) > 1:
        print("\nlarge-tick cohort spread (the n=1 gap this run closes):")
        for col in ["mm_pnl", "mm_at_touch_pct", "mm_peak_abs_inventory", "exe_fill_rate_pct"]:
            v = msft_rows[col].tolist()
            print(f"  {col:24s} mean={statistics.mean(v):10,.3f}  "
                  f"sd={statistics.stdev(v):9,.3f}  min={min(v):10,.3f}  max={max(v):10,.3f}")

    dest = ROOT / "results" / "msft_cohort.csv"
    out.to_csv(dest, index=False)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
