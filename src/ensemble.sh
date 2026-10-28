# Build ensembles for each position on every fold in config.py (VALID_FOLDS, then TEST_FOLDS).
# Run folds.sh first: the top-K models of each position are ranked from its validation results.
#
# Each method is run on two sets of base models:
#   ens_<method>_base   the BASE_MODELS below, the same for every position
#   ens_<method>_top<K> the TOP_K base models with the lowest validation MAE for that position
# The "top" method instead picks its TOP_K models inside each fold (see ensemble.py).
#
# Usage:
#   bash ensemble.sh                 # mean
#   bash ensemble.sh mean stack top
#   TOP_K=4 BASE_MODELS="ridge svr cat" bash ensemble.sh
set -e
cd "$(dirname "$0")"

METHODS=${*:-"mean"}
BASE_MODELS=${BASE_MODELS:-"ridge hgb lgbm xgb svr"}
TOP_K=${TOP_K:-3}
POSITIONS=$(python -c "import config; print(*config.POSITIONS)")
FOLDS=$(python -c "import config; print(*config.VALID_FOLDS, *config.TEST_FOLDS)")
CANDIDATES=$(python -c "import model_dispatcher as md; print(*[m for m in md.models_gk if m != 'dummy'])")

# Python sometimes segfaults (exit 139) after training lightgbm/xgboost; rerun those.
run() {
  local status
  for try in 1 2 3; do
    status=0
    python "$@" || status=$?
    [ $status -ne 139 ] && break
    echo "--- segfault, retry $try: $* ---"
  done
  return $status
}

for pos in $POSITIONS; do
  TOP_MODELS=$(python compare.py --rank "$pos" "$TOP_K")
  if [ -z "$TOP_MODELS" ]; then
    echo "No validation results for $pos. Run folds.sh first." >&2
    exit 1
  fi
  echo "=== $pos: base = $BASE_MODELS | top$TOP_K = $TOP_MODELS ==="
  for method in $METHODS; do
    for fold in $FOLDS; do
      if [ "$method" = top ]; then
        run ensemble.py --fold "$fold" --pos "$pos" --method top --top-k "$TOP_K" \
          --models $CANDIDATES --name "ens_top${TOP_K}_per_fold"
      else
        run ensemble.py --fold "$fold" --pos "$pos" --method "$method" \
          --models $BASE_MODELS --name "ens_${method}_base"
        run ensemble.py --fold "$fold" --pos "$pos" --method "$method" \
          --models $TOP_MODELS --name "ens_${method}_top${TOP_K}"
      fi
    done
  done
done
