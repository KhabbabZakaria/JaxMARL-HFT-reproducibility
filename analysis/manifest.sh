#!/bin/bash
# Print the seed / budget / entropy / ticker set of every archived run, read back
# out of the config each trainer echoes at the top of its log. This is the source
# of truth for the RUNS table in make_report.py.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for f in results/raw_logs/*.log.gz; do
    name=$(basename "$f" .log.gz)
    cfg=$(gzcat "$f" 2>/dev/null || zcat "$f")
    seed=$(echo "$cfg" | grep -o "'SEED': [0-9]*" | head -1 | grep -o '[0-9]*')
    tt=$(echo "$cfg" | grep -o "'TOTAL_TIMESTEPS': [0-9.e+]*" | head -1 | cut -d' ' -f2)
    ent=$(echo "$cfg" | grep -o "'ENT_COEF': \[[^]]*\]" | head -1 | cut -d: -f2)
    stock=$(echo "$cfg" | grep -o "stock='[A-Z,]*'" | head -1 | cut -d"'" -f2)
    printf "%-28s seed=%-3s timesteps=%-10s ENT_COEF=%-14s tickers=%s\n" \
           "$name" "$seed" "$tt" "$ent" "$stock"
done
