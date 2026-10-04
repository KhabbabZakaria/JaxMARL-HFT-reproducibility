#!/bin/bash
# Large-tick replication: MSFT alone, three seeds in parallel.
#
# The original large-tick run was INTC,MSFT on a single seed. INTC's level-1 book
# file was not retained, so this cohort is MSFT-only -- the instrument whose spread
# and queue depth were actually measured (1-tick spread 75.8% of the time, median
# 10,907 shares at the touch). It is a fresh condition, not a replication of the
# INTC,MSFT pair, and must be reported as such.
#
# 4e6 timesteps = 975 updates, the budget every other number in the study uses.
# Three parallel runs of this size took ~24.5h wall clock on this machine.
#
# Usage: ./train_msft_seeds.sh [SEED ...]     (default: 2 7 13)
set -e

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f .env ] && { set -a; . ./.env; set +a; }
[ -f venv/bin/activate ] && source venv/bin/activate
export PYTHONPATH=$(pwd):$PYTHONPATH

SEEDS=${@:-2 7 13}
NOSLEEP=$(command -v caffeinate >/dev/null && echo "caffeinate -i" || true)

for SEED in $SEEDS; do
    LOG="run_MSFT_seed${SEED}.log"
    nohup $NOSLEEP python3 gymnax_exchange/jaxrl/MARL/ippo_rnn_JAXMARL.py \
        --config-name=ippo_rnn_JAXMARL_2player \
        ENV_CONFIG=config/env_configs/2_player_fq_fqc_msft.json \
        WANDB_MODE=disabled \
        NUM_ENVS=64 \
        NUM_STEPS=64 \
        TOTAL_TIMESTEPS=4000000 \
        GRU_HIDDEN_DIM=64 \
        FC_DIM_SIZE=64 \
        SEED=$SEED \
        > "$LOG" 2>&1 &
    echo "seed $SEED -> PID $!  log=$LOG"
done

echo
echo "watch:  grep -c 'completed' run_MSFT_seed*.log"
echo "stop :  pkill -f ippo_rnn_JAXMARL"
