#!/bin/bash
# Camera-ready LOMO quota-strength sweep (k = 3, 0, 1, 2, 5) with M3 refit in every rotation.
# The first run (k=3) fits and caches the nine rotation fits; the others reuse the cache.
set -e
cd "$(dirname "$0")/.."
for k in 3 0 1 2 5; do
  python scripts/run_lomo_full.py --models m3 --m3-init median --lr-decay 0.1 \
    --steps 3000 --seeds 2 --sizes 100,150,200,300 --min-k $k --dump-raw \
    --cache results/camera_ready/lomo_fits --out results/camera_ready/lomo_m3_k${k}_raw.json
done
