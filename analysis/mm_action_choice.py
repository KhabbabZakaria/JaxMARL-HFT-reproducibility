"""Where the market maker actually places its quote, and what it converged on.

The mechanism behind the ticker-regime finding lives in the `fixed_quants` action
space (mm_env.py::_getActionMsgs_fixedQuant). Each action picks an offset for the
bid and the ask, measured in multiples of half the current spread and applied
OUTWARD FROM THE TOUCH:

    bid_price = best_bid - bid_offset * half_spread
    ask_price = best_ask + ask_offset * half_spread

So offset 0 quotes exactly at the touch and every other offset quotes behind it.
There is no action that quotes inside the spread. Note that `spread_multiplier`
plays no part here -- it belongs to the `spread_skew` action space, not this one.

Quote size is `fixed_quant_value` (bid_quants[action] is a 1/0 on-off multiplier),
which is 1 share in this study's config.

Usage:  python analysis/mm_action_choice.py
"""
import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
MET = ROOT / "results" / "metrics"
BUDGET, TAIL = 975, 97

# mm_env.py:1009-1010, for sell_buy_all_option == False.
BID_OFFSETS = [0, 1, 2, 3, 4, 0, 2, 5, 1, 0]
ASK_OFFSETS = [0, 1, 2, 3, 4, 2, 0, 1, 5, 0]
# mm_env.py:1011-1012 -- action 9 posts nothing at all.
POSTS = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]

RUNS = {"run_AAPL_AMZN_GOOG": "small-tick (AAPL,AMZN,GOOG)",
        "run_INTC_MSFT": "large-tick (INTC,MSFT)"}


def main():
    print("action  bid_offset  ask_offset  posts?   placement")
    for a in range(10):
        if not POSTS[a]:
            place = "posts nothing"
        elif BID_OFFSETS[a] == 0 and ASK_OFFSETS[a] == 0:
            place = "both sides AT the touch"
        elif BID_OFFSETS[a] == 0 or ASK_OFFSETS[a] == 0:
            place = "one side at the touch, one behind"
        else:
            place = "both sides behind the touch"
        print(f"  {a:<6d}{BID_OFFSETS[a]:>8d}{ASK_OFFSETS[a]:>12d}{'yes' if POSTS[a] else 'no':>8s}   {place}")

    rows = []
    print("\nmost-used action over the final 10% of the common budget:")
    for run, label in RUNS.items():
        d = pd.read_csv(MET / f"{run}.csv").head(BUDGET).tail(TAIL)
        vc = d["mm_top_action"].value_counts(normalize=True).sort_values(ascending=False) * 100
        share = ", ".join(f"a{int(k)}={v:.0f}%" for k, v in vc.head(3).items())
        at_touch = sum(v for k, v in vc.items()
                       if BID_OFFSETS[int(k)] == 0 and ASK_OFFSETS[int(k)] == 0 and POSTS[int(k)])
        print(f"  {label:28s} {share:34s}  at-touch {at_touch:5.1f}% of updates")
        rows.append({"run": run, "regime": label,
                     "top_action_mode": int(vc.index[0]),
                     "top_action_mode_share_pct": round(float(vc.iloc[0]), 1),
                     "pct_updates_quoting_at_touch": round(float(at_touch), 1)})

    out = ROOT / "results" / "mm_action_choice.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
