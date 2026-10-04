#!/bin/bash
# Entropy-coefficient experiment. Usage: ./train_ent.sh <SEED> <ENT_EXE> [TOTAL_TIMESTEPS]
#
# ENT_COEF is per agent type: [MM, EXE]. MM stays at the 0.01 baseline; only the
# execution agent's exploration pressure is varied, since that is the agent that
# gets trapped on action 0 (No trade).
set -e

SEED=${1:?usage: ./train_ent.sh <SEED> <ENT_EXE> [TOTAL_TIMESTEPS]}
ENT=${2:?usage: ./train_ent.sh <SEED> <ENT_EXE> [TOTAL_TIMESTEPS]}
TT=${3:-4000000}

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Machine-specific paths and any W&B credentials. The Python side reads .env too,
# so this is only needed for variables the shell itself passes through.
# set -a exports everything .env defines; sourcing keeps quoting intact, so paths
# containing spaces survive (a bare `export $(cat .env)` would split them).
[ -f .env ] && { set -a; . ./.env; set +a; }

# Activate a local venv if there is one; otherwise use whatever python3 is active.
[ -f venv/bin/activate ] && source venv/bin/activate
export PYTHONPATH=$(pwd):$PYTHONPATH

LOG="run_ent${ENT}_seed${SEED}.log"

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
    "ENT_COEF=[0.01,$ENT]" \
    > "$LOG" 2>&1 &

echo "seed $SEED  ENT_COEF(EXE)=$ENT  -> PID $!  log=$LOG"
