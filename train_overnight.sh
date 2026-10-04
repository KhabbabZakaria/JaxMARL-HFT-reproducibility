#!/bin/bash
# Overnight IPPO run: 1 market maker + 1 execution agent, 3 tickers, CPU.
# NUM_ENVS=64 is the measured throughput peak on this Mac; larger values thrash.
set -e

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Machine-specific paths and any W&B credentials. The Python side reads .env too,
# so this is only needed for variables the shell itself passes through.
# set -a exports everything .env defines; sourcing keeps quoting intact, so paths
# containing spaces survive (a bare `export $(cat .env)` would split them).
[ -f .env ] && { set -a; . ./.env; set +a; }

# Activate a local venv if there is one; otherwise use whatever python3 is active.
[ -f venv/bin/activate ] && source venv/bin/activate
export PYTHONPATH=$(pwd):$PYTHONPATH

# Name the log after the tickers in the env config so successive experiments
# don't overwrite each other's results.
TICKERS=$(python3 -c "import json;print(json.load(open('config/env_configs/2_player_fq_fqc.json'))['world_config']['stock'].replace(',','_'))")
LOG="run_${TICKERS}.log"

# caffeinate -i keeps the Mac awake; a sleeping laptop suspends the run.
NOSLEEP=$(command -v caffeinate >/dev/null && echo "caffeinate -i" || true)

nohup $NOSLEEP python3 gymnax_exchange/jaxrl/MARL/ippo_rnn_JAXMARL.py \
    --config-name=ippo_rnn_JAXMARL_2player \
    WANDB_MODE=disabled \
    NUM_ENVS=64 \
    NUM_STEPS=64 \
    TOTAL_TIMESTEPS=10000000 \
    GRU_HIDDEN_DIM=64 \
    FC_DIM_SIZE=64 \
    > "$LOG" 2>&1 &

echo "started, PID $!  ->  $LOG"
echo "watch with:  tail -f $LOG"
echo "activity:    grep -E '^  \[' $LOG | tail -20"
