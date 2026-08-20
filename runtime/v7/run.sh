#!/usr/bin/env bash
# One V7 command: fast closure check, then full resident two-rank training.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RANK=${V7_RANK:?Set V7_RANK=0 on Spark-1 or V7_RANK=1 on Spark-3}
PYTHON=${V7_PYTHON:-/home/dnola/humming_env/bin/python}
MODEL_ROOT=${V7_MODEL_ROOT:?Set V7_MODEL_ROOT to the local regular model directory}
MEMBER_ROSTER=${V7_MEMBER_ROSTER:?Set V7_MEMBER_ROSTER to the all-43 selected-wire roster}
MEMBER_ROSTER_SHA256=${V7_MEMBER_ROSTER_SHA256:?Set V7_MEMBER_ROSTER_SHA256 to its pinned SHA-256}
LP4_ROOT=${V7_LP4_ROOT:-$ROOT/vendor}
LP4_PACK=${V7_LP4_PACK:?Set V7_LP4_PACK to the local LP4 pack directory}
LP4_MANIFEST=${V7_LP4_MANIFEST:?Set V7_LP4_MANIFEST to the LP4 manifest JSON}
LP4_SELECTION=${V7_LP4_SELECTION:-}
DELTA_DIR=${V7_DELTA_DIR:?Set V7_DELTA_DIR to a directory containing DELTA_PACK.COMPLETE}
VQ3B_DIR=${V7_VQ3B_DIR:-$MODEL_ROOT}
CORPUS=${V7_CORPUS:?Set V7_CORPUS to windows_ds4_TRAIN.json}
TEACH=${V7_TEACH:?Set V7_TEACH to the published teacher bank}
OUTPUT=${V7_OUTPUT:-$ROOT/output}
RUN_ROOT=${V7_RUN_ROOT:-$OUTPUT/rank${RANK}}
MASTER_ADDR=${V7_MASTER_ADDR:-192.168.200.1}
MASTER_PORT=${V7_MASTER_PORT:-29672}
TASK_ID=${V7_TASK_ID:-v7-unified}
CLAIM_OWNER=${V7_CLAIM_OWNER:-v7}
UPDATES=${V7_UPDATES:-64}

mkdir -p "$OUTPUT" "$RUN_ROOT" "$OUTPUT/extensions"
export V7_VENDOR_ROOT="$ROOT/vendor"
export V7_LP4_ROOT="$LP4_ROOT"
export V7_LP4_PACK="$LP4_PACK"
export V7_LP4_MANIFEST="$LP4_MANIFEST"
export V7_LP4_SELECTION="$LP4_SELECTION"
export BANANA_SMASHER_PUBLIC_SRC="$ROOT/vendor/site"
export BR_MANIFEST="$LP4_MANIFEST"
export BR_DELTA_DIR="$DELTA_DIR"
export BR_VQ3B_DIR="$VQ3B_DIR"
export BR_CORPUS="$CORPUS"
export BR_TEACH="$TEACH"
export BR_TRAIN="20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83"
export BR_PROBE="20"
export TORCH_EXTENSIONS_DIR="$OUTPUT/extensions"

"$PYTHON" "$ROOT/contract_smoke.py" \
  --root "$ROOT" \
  --model-root "$MODEL_ROOT" \
  --member-roster "$MEMBER_ROSTER" \
  --expected-member-roster-sha256 "$MEMBER_ROSTER_SHA256" \
  --manifest "$LP4_MANIFEST" \
  --delta-dir "$DELTA_DIR" \
  --vq3b-dir "$VQ3B_DIR" \
  --corpus "$CORPUS" \
  --teacher "$TEACH" \
  --lp4-pack "$LP4_PACK" \
  ${LP4_SELECTION:+--lp4-selection "$LP4_SELECTION"} \
  --compile-native | tee "$OUTPUT/closure-rank${RANK}.json"

exec "$PYTHON" -u "$ROOT/runner/fast_two_node_v7.py" \
  --rank "$RANK" \
  --run-root "$RUN_ROOT" \
  --asset-root "$ROOT" \
  --model-root "$MODEL_ROOT" \
  --member-roster "$MEMBER_ROSTER" \
  --expected-member-roster-sha256 "$MEMBER_ROSTER_SHA256" \
  --fresh-u0 \
  --expected-claim-owner "$CLAIM_OWNER" \
  --task-id "$TASK_ID" \
  --master-addr "$MASTER_ADDR" \
  --master-port "$MASTER_PORT" \
  --updates "$UPDATES" \
  --split-layer 21
