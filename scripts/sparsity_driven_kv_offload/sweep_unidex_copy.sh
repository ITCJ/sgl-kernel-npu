#!/usr/bin/env bash
# Compare row-partitioned and column-partitioned indexed copies.
#
# Usage:
#   bash scripts/sparsity_driven_kv_offload/sweep_unidex_copy.sh
#
# Override params via environment:
#   BATCH_SIZES="1 4 16" MAX_RUNNING_REQUESTS=16 BLOCK_DIMS="24 48" \
#     bash scripts/sparsity_driven_kv_offload/sweep_unidex_copy.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_SCRIPT="${REPO_ROOT}/benchmark/sparsity_driven_kv_offload/bench_unidex_copy.py"

# ---------- tunable parameter lists ----------
HIT_RATES="${HIT_RATES:-0.5}"
BLOCK_DIMS="${BLOCK_DIMS:-8 16 24 32 48}"
TOPK_LIST="${TOPK_LIST:-2048}"
BATCH_SIZES="${BATCH_SIZES:-1 4 16}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-16}"

# fixed arguments
SRC_ROWS="${SRC_ROWS:-131072}"
DST_ROWS="${DST_ROWS:-0}"
SRC_IDX_MODE="${SRC_IDX_MODE:-random}"
DST_IDX_MODE="${DST_IDX_MODE:-arange}"
DTYPE="${DTYPE:-float16}"
HEAD_NUM="${HEAD_NUM:-1}"
HEAD_DIM="${HEAD_DIM:-576}"
TOKEN_BYTES="${TOKEN_BYTES:-1152}"
WARMUP="${WARMUP:-5}"
PERF_ITERS="${PERF_ITERS:-50}"
ACC_ITERS="${ACC_ITERS:-1}"
SEED="${SEED:-20260609}"

cd "${REPO_ROOT}"

for batch_size in ${BATCH_SIZES}; do
  for topk in ${TOPK_LIST}; do
    for block_dim in ${BLOCK_DIMS}; do
      for hit_rate in ${HIT_RATES}; do
        echo "============================================================"
        echo "uindex copy bench: batch=${batch_size}/${MAX_RUNNING_REQUESTS} topk=${topk} block_dim=${block_dim} hit_rate=${hit_rate}"
        echo "============================================================"
        python3 "${BENCH_SCRIPT}" \
          --directions d2d \
          --baselines unidex uindex_optimized \
          --batch-size "${batch_size}" \
          --max-running-requests "${MAX_RUNNING_REQUESTS}" \
          --topk "${topk}" \
          --src-rows "${SRC_ROWS}" \
          --dst-rows "${DST_ROWS}" \
          --src-index-mode "${SRC_IDX_MODE}" \
          --dst-index-mode "${DST_IDX_MODE}" \
          --hit-rate "${hit_rate}" \
          --dtype "${DTYPE}" \
          --head-num "${HEAD_NUM}" \
          --head-dim "${HEAD_DIM}" \
          --token-bytes "${TOKEN_BYTES}" \
          --block-dim "${block_dim}" \
          --warmup "${WARMUP}" \
          --perf-iters "${PERF_ITERS}" \
          --accuracy-iters "${ACC_ITERS}" \
          --seed "${SEED}"
        echo ""
      done
    done
  done
done
