#!/bin/bash
# Seed-replication run. Usage: ./train_seed.sh <SEED> [TOTAL_TIMESTEPS]
#
# Everything except SEED matches the completed AAPL,AMZN,GOOG run, so the only
# variable is the RNG. 4e6 timesteps = 976 updates, which covers the whole
# interesting region: both prior runs had fully converged before update ~1000.
set -e

SEED=${1:?usage: ./train_seed.sh <SEED> [TOTAL_TIMESTEPS]}
TT=${2:-4000000}

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Machine-specific paths and any W&B credentials. The Python side reads .env too,
# so this is only needed for variables the shell itself passes through.
# set -a exports everything .env defines; sourcing keeps quoting intact, so paths
# containing spaces survive (a bare `export $(cat .env)` would split them).
[ -f .env ] && { set -a; . ./.env; set +a; }

# Activate a local venv if there is one; otherwise use whatever python3 is active.
[ -f venv/bin/activate ] && source venv/bin/activate
export PYTHONPATH=$(pwd):$PYTHONPATH

TICKERS=$(python3 -c "import json;print(json.load(open('config/env_configs/2_player_fq_fqc.json'))['world_config']['stock'].replace(',','_'))")
LOG="run_${TICKERS}_seed${SEED}.log"

NOSLEEP=$(command -v caffeinate >/dev/null && echo "caffeinate -i" || true)

nohup $NOSLEEP python3 gymnax_exchange/jaxrl/MARL/ippo_rnn_JAXMARL.py \
    --config-name=ippo_rnn_JAXMARL_2player \
    WANDB_MODE=disabled \
    NUM_ENVS=64 \
    NUM_STEPS=64 \
    TOTAL_TIMESTEPS=$TT \
    GRU_HIDDEN_DIM=64 \
    FC_DIM_SIZE=64 \
    SEED=$SEED \
    > "$LOG" 2>&1 &

echo "seed $SEED -> PID $!  log=$LOG"
