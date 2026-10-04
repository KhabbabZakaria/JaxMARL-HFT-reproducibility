"""Build the matched-budget comparison tables and figures from results/metrics/*.csv.

Every table here truncates each run to BUDGET updates before comparing. The seed
runs were given 4e6 timesteps (976 updates) while the two exploratory runs got
1e7, so comparing final values directly would confound seed with training length.

Usage:  python analysis/make_report.py
"""
import pathlib
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BUDGET = 975        # updates common to every run (4e6 steps / (64 envs * 64 steps))
EPISODE_STEPS = 64  # see extract_metrics.EPISODE_STEPS
TASK_SIZE = 600
TAIL = 0.1          # fraction of updates averaged to describe converged behaviour

ROOT = pathlib.Path(__file__).resolve().parent.parent
MET, FIG = ROOT / "results" / "metrics", ROOT / "results" / "figures"

# run -> (seed, total_timesteps, ENT_COEF[EXE], tickers). Read off the config
# echoed at the top of each raw log; regenerate with analysis/manifest.sh.
RUNS = {
    "run_AAPL_AMZN_GOOG":        (2,  10_000_000, 0.01, "AAPL,AMZN,GOOG"),
    "run_AAPL_AMZN_GOOG_seed7":  (7,   4_000_000, 0.01, "AAPL,AMZN,GOOG"),
    "run_AAPL_AMZN_GOOG_seed13": (13,  4_000_000, 0.01, "AAPL,AMZN,GOOG"),
    "run_AAPL_AMZN_GOOG_seed42": (42,  4_000_000, 0.01, "AAPL,AMZN,GOOG"),
    "run_ent0.05_seed7":         (7,   4_000_000, 0.05, "AAPL,AMZN,GOOG"),
    "run_ent0.10_seed7":         (7,   4_000_000, 0.10, "AAPL,AMZN,GOOG"),
    "run_INTC_MSFT":             (2,  10_000_000, 0.01, "INTC,MSFT"),
}
SEED_RUNS = ["run_AAPL_AMZN_GOOG", "run_AAPL_AMZN_GOOG_seed7",
             "run_AAPL_AMZN_GOOG_seed13", "run_AAPL_AMZN_GOOG_seed42"]
ENT_RUNS = ["run_AAPL_AMZN_GOOG_seed7", "run_ent0.05_seed7", "run_ent0.10_seed7"]


def load(run, budget=BUDGET):
    df = pd.read_csv(MET / f"{run}.csv")
    return df.head(budget)


def tail_stats(run, budget=BUDGET):
    df = load(run, budget)
    k = max(1, int(len(df) * TAIL))
    t = df.tail(k)
    doom = t["exe_doom"].mean()
    seed, tt, ent, tick = RUNS[run]
    return {
        "run": run, "seed": seed, "timesteps": tt, "ent_coef_exe": ent,
        "tickers": tick, "updates_used": len(df), "tail_updates": k,
        "reward_mm": t["reward_mm"].mean(),
        "reward_exe": t["reward_exe"].mean(),
        "mm_pnl": t["mm_pnl"].mean(),
        "mm_inventory": t["mm_inv"].mean(),
        "mm_top_action_pct": t["mm_top_action_pct"].mean(),
        "exe_doom_mean": doom,
        "exe_unfilled_est": doom * EPISODE_STEPS,
        "exe_fill_rate_est": 1 - doom * EPISODE_STEPS / TASK_SIZE,
        "exe_quant_left_mean": t["exe_left"].mean(),
        "exe_top_action_pct": t["exe_top_action_pct"].mean(),
    }


def spread_row(label, values):
    m, s = statistics.mean(values), statistics.stdev(values)
    return {"metric": label, "mean": m, "std": s, "min": min(values),
            "max": max(values), "spread": max(values) - min(values),
            "spread_pct_of_mean": abs((max(values) - min(values)) / m) * 100 if m else float("nan")}


def main():
    FIG.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame([tail_stats(r) for r in RUNS])
    summary.to_csv(ROOT / "results" / "summary_matched.csv", index=False)
    print("== matched-budget summary (first 975 updates of every run) ==")
    print(summary[["run", "seed", "ent_coef_exe", "tickers", "reward_mm", "reward_exe",
                   "exe_unfilled_est", "exe_fill_rate_est", "mm_pnl"]].to_string(index=False))

    # --- seed variance ------------------------------------------------------
    seed_stats = [tail_stats(r) for r in SEED_RUNS]
    var = pd.DataFrame([
        spread_row("reward_mm", [s["reward_mm"] for s in seed_stats]),
        spread_row("reward_exe", [s["reward_exe"] for s in seed_stats]),
        spread_row("exe_unfilled_est", [s["exe_unfilled_est"] for s in seed_stats]),
        spread_row("mm_pnl", [s["mm_pnl"] for s in seed_stats]),
        spread_row("mm_inventory", [s["mm_inventory"] for s in seed_stats]),
    ])
    var.to_csv(ROOT / "results" / "seed_variance.csv", index=False)
    print("\n== seed variance across seeds 2/7/13/42 at identical budget ==")
    print(var.to_string(index=False))

    # --- entropy sweep vs seed noise ---------------------------------------
    ent_stats = [tail_stats(r) for r in ENT_RUNS]
    ent_spread = max(s["reward_exe"] for s in ent_stats) - min(s["reward_exe"] for s in ent_stats)
    seed_spread = max(s["reward_exe"] for s in seed_stats) - min(s["reward_exe"] for s in seed_stats)
    print(f"\n== entropy sweep (seed 7) ==")
    for s in ent_stats:
        print(f"  ENT_COEF[EXE]={s['ent_coef_exe']:.2f}  reward_exe={s['reward_exe']:+.3f}"
              f"  unfilled={s['exe_unfilled_est']:5.0f}u  fill={s['exe_fill_rate_est']*100:4.1f}%")
    print(f"  entropy-induced spread in reward_exe : {ent_spread:.3f}")
    print(f"  seed-induced    spread in reward_exe : {seed_spread:.3f}")
    print(f"  -> hyperparameter effect is {ent_spread/seed_spread:.2f}x the seed noise")

    # --- figures ------------------------------------------------------------
    def smooth(s, w=25):
        return s.rolling(w, min_periods=1).mean()

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for r in SEED_RUNS:
        df = load(r)
        ax[0].plot(df["update"], smooth(df["reward_mm"]), label=f"seed {RUNS[r][0]}", lw=1.3)
        ax[1].plot(df["update"], smooth(df["reward_exe"]), label=f"seed {RUNS[r][0]}", lw=1.3)
    ax[0].set_title("Market-making agent reward"); ax[1].set_title("Execution agent reward")
    for a in ax:
        a.set_xlabel("update"); a.set_ylabel("mean reward"); a.legend(fontsize=8); a.grid(alpha=.3)
    fig.suptitle("Four seeds, identical config and budget (25-update moving average)")
    fig.tight_layout(); fig.savefig(FIG / "seed_variance.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for r in SEED_RUNS:
        df = load(r)
        ax[0].plot(df["update"], smooth(df["exe_doom"] * EPISODE_STEPS),
                   label=f"seed {RUNS[r][0]}", lw=1.3)
        ax[1].plot(df["update"], smooth(df["exe_top_action_pct"]),
                   label=f"seed {RUNS[r][0]}", lw=1.3)
    ax[0].axhline(TASK_SIZE, ls="--", c="k", lw=.8)
    ax[0].text(10, TASK_SIZE * .95, f"whole task ({TASK_SIZE}u) unfilled", fontsize=7)
    ax[0].set_ylabel(f"units force-liquidated at episode end (of {TASK_SIZE})")
    ax[0].set_title("Execution shortfall")
    ax[1].set_ylabel("share of steps taking the single most-used action (%)")
    ax[1].set_title("Execution policy concentration")
    for a in ax:
        a.set_xlabel("update"); a.legend(fontsize=8); a.grid(alpha=.3)
    fig.suptitle("The reward curve hides a large spread in task completion")
    fig.tight_layout(); fig.savefig(FIG / "execution_behaviour.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    lo = min(s["reward_exe"] for s in seed_stats); hi = max(s["reward_exe"] for s in seed_stats)
    ax[0].axhspan(lo, hi, color="grey", alpha=.25,
                  label="range over 4 seeds, ENT fixed at 0.01")
    for s in ent_stats:
        ax[0].scatter(s["ent_coef_exe"], s["reward_exe"], s=70, zorder=3)
        ax[0].annotate(f"{s['reward_exe']:+.2f}", (s["ent_coef_exe"], s["reward_exe"]),
                       textcoords="offset points", xytext=(8, -3), fontsize=8)
    ax[0].set_xlabel("ENT_COEF (execution agent)"); ax[0].set_ylabel("mean reward, final 10% of updates")
    ax[0].set_title(f"10x entropy change moves reward {ent_spread:.2f};"
                    f" seed alone moves it {seed_spread:.2f}")
    ax[0].set_xlim(0, 0.12); ax[0].legend(fontsize=8, loc="lower left"); ax[0].grid(alpha=.3)
    ax[1].scatter([s["ent_coef_exe"] for s in ent_stats],
                  [s["exe_unfilled_est"] for s in ent_stats], s=70, c="tab:red", zorder=3)
    for s in ent_stats:
        ax[1].annotate(f"{s['exe_unfilled_est']:.0f}u", (s["ent_coef_exe"], s["exe_unfilled_est"]),
                       textcoords="offset points", xytext=(8, -3), fontsize=8)
    ax[1].set_xlabel("ENT_COEF (execution agent)"); ax[1].set_ylabel("units unfilled at episode end")
    ax[1].set_title("...while task completion improves monotonically")
    ax[1].set_xlim(0, 0.12); ax[1].grid(alpha=.3)
    fig.suptitle("Reward and task completion rank the same runs differently")
    fig.tight_layout(); fig.savefig(FIG / "entropy_sweep.png", dpi=140); plt.close(fig)

    # Truncated at BUDGET like every other comparison, so what the figure shows is
    # exactly what the table measures. Both runs continue to 2,440 updates with the
    # gap unchanged; that is stated in the text rather than drawn, because a curve
    # extending past the measurement point invites the reader to read values off it
    # that no table reports.
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for r, c in (("run_AAPL_AMZN_GOOG", "tab:blue"), ("run_INTC_MSFT", "tab:orange")):
        df = load(r)
        ax[0].plot(df["update"], smooth(df["reward_mm"]), c=c, label=RUNS[r][3], lw=1.3)
        ax[1].plot(df["update"], smooth(df["mm_pnl"]), c=c, label=RUNS[r][3], lw=1.3)
    ax[0].set_ylabel("market-making reward"); ax[1].set_ylabel("market-making PnL (ticks)")
    ax[0].set_title("Reward"); ax[1].set_title("PnL")
    for a in ax:
        a.set_xlabel("update"); a.legend(fontsize=8); a.grid(alpha=.3)
    fig.suptitle("Same code, same seed, same day — two ticker sets "
                 "(first 975 updates, the budget every reported number uses)")
    fig.tight_layout(); fig.savefig(FIG / "ticker_regime.png", dpi=140); plt.close(fig)

    print(f"\nfigures -> {FIG}")


if __name__ == "__main__":
    main()
