"""Turn the raw training stdout logs into tidy per-update CSVs.

The trainer prints two kinds of line we care about, once per update:

    avg_reward_MM  <float>          # mean reward of the market-making agent
    avg_reward_EXE <float>          # mean reward of the execution agent
      [ MM] inv=... pnl=... top_action=7(11%)
      [EXE] doom=... left=... top_action=11(34%)

The `[MM]/[EXE]` lines come from the stdout patch in ippo_rnn_JAXMARL.py; the
avg_reward lines are upstream. Everything else in the log is Hydra/orbax noise.

Usage:  python analysis/extract_metrics.py [--logs DIR] [--out DIR]
"""
import argparse
import csv
import gzip
import pathlib
import re
import statistics

# label -> column name, matching the stdout_fields table in the trainer.
AGENT_FIELDS = ("doom", "left", "inv", "mktshr", "bidq", "askq", "pnl")
NUM_RE = re.compile(r"(\w+)=\s*([-+]?[0-9]*\.?[0-9]+)")
TOP_RE = re.compile(r"top_action=(\d+)\((\d+)%\)")


def _open(path):
    """Read either a plain .log or the committed .log.gz."""
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", errors="ignore")
    return open(path, errors="ignore")


def parse_log(path):
    """Yield one dict per completed update.

    A record is flushed when the [EXE] line arrives, because the trainer prints
    the agents in a fixed order (MM then EXE) and EXE is last. Updates whose
    output was cut off mid-way by an aborted run are therefore dropped.
    """
    rows, cur = [], {}
    with _open(path) as fh:
        for line in fh:
            if line.startswith("avg_reward_MM"):
                cur["reward_mm"] = float(line.split()[1])
            elif line.startswith("avg_reward_EXE"):
                cur["reward_exe"] = float(line.split()[1])
            elif line.startswith("  [ MM]") or line.startswith("  [EXE]"):
                prefix = "mm_" if "[ MM]" in line else "exe_"
                for key, val in NUM_RE.findall(line):
                    if key in AGENT_FIELDS:
                        cur[prefix + key] = float(val)
                top = TOP_RE.search(line)
                if top:
                    cur[prefix + "top_action"] = int(top.group(1))
                    cur[prefix + "top_action_pct"] = int(top.group(2))
                if prefix == "exe_" and "reward_exe" in cur:
                    cur["update"] = len(rows) + 1
                    rows.append(cur)
                    cur = {}
    return rows


COLUMNS = ["update", "reward_mm", "reward_exe",
           "mm_inv", "mm_pnl", "mm_bidq", "mm_askq", "mm_mktshr",
           "mm_top_action", "mm_top_action_pct",
           "exe_doom", "exe_left", "exe_top_action", "exe_top_action_pct"]

# Episode length in steps (world_config.episode_time with ep_type=fixed_steps).
# doom_quant is logged as a mean over all steps of the rollout but is non-zero
# only on the single terminal step, so the quantity actually force-liquidated at
# the end of an episode is recovered by multiplying the mean back up.
EPISODE_STEPS = 64
TASK_SIZE = 600  # Execution_EnvironmentConfig.task_size


def summarise(rows, tail_frac=0.1):
    """Mean of each metric over the last `tail_frac` of updates.

    Averaging a tail rather than taking the final update matters here: the
    per-update means are noisy enough that a single update is not a stable
    estimate of converged behaviour.
    """
    if not rows:
        return {}
    k = max(1, int(len(rows) * tail_frac))
    tail = rows[-k:]

    def mean(col):
        vals = [r[col] for r in tail if col in r]
        return statistics.mean(vals) if vals else float("nan")

    doom = mean("exe_doom")
    out = {"updates": len(rows), "tail_updates": k}
    for col in COLUMNS[1:]:
        out[col] = mean(col)
    out["exe_unfilled_est"] = doom * EPISODE_STEPS
    out["exe_fill_rate_est"] = 1.0 - (doom * EPISODE_STEPS) / TASK_SIZE
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="results/raw_logs",
                    help="directory holding run_*.log or run_*.log.gz")
    ap.add_argument("--out", default="results/metrics")
    ap.add_argument("--tail-frac", type=float, default=0.1)
    args = ap.parse_args()

    log_dir, out_dir = pathlib.Path(args.logs), pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    logs = sorted(list(log_dir.glob("run_*.log")) + list(log_dir.glob("run_*.log.gz")))
    if not logs:
        raise SystemExit(f"no run_*.log[.gz] found in {log_dir}")

    summaries = []
    for log in logs:
        rows = parse_log(log)
        name = log.name.split(".log")[0]
        with open(out_dir / f"{name}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        summaries.append({"run": name, **summarise(rows, args.tail_frac)})
        print(f"{name:34s} {len(rows):5d} updates -> {name}.csv")

    keys = ["run"] + [k for k in summaries[0] if k != "run"]
    with open(pathlib.Path(args.out).parent / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(summaries)
    print(f"\nwrote {pathlib.Path(args.out).parent / 'summary.csv'}")


if __name__ == "__main__":
    main()
