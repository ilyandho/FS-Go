# Build ensembles for each position at every selection point in config.VALID_FOLDS.
# Each one is a candidate model next to the base models from folds.sh.
#
# Methods:
#   mean, median, stack  combine BASE_MODELS (the same for every position), saved as ens_<method>_base
#   top                  average the TOP_K models that score best on the 3 gameweeks before the
#                        selection point, chosen from every base model, saved as ens_top<K>
#
# Usage:
#   bash ensemble.sh                 # mean and top
#   bash ensemble.sh mean stack
#   TOP_K=4 BASE_MODELS="ridge svr cat" bash ensemble.sh
set -e
cd "$(dirname "$0")"

METHODS=${*:-"mean top"}
BASE_MODELS=${BASE_MODELS:-"ridge hgb lgbm xgb svr"}
TOP_K=${TOP_K:-3}
POSITIONS=$(python -c "import config; print(*config.POSITIONS)")
FOLDS=$(python -c "import config; print(*config.VALID_FOLDS)")
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
  for method in $METHODS; do
    for fold in $FOLDS; do
      if [ "$method" = top ]; then
        run ensemble.py --fold "$fold" --pos "$pos" --method top --top-k "$TOP_K" \
          --models $CANDIDATES --name "ens_top${TOP_K}"
      else
        run ensemble.py --fold "$fold" --pos "$pos" --method "$method" \
          --models $BASE_MODELS --name "ens_${method}_base"
      fi
    done
  done
done
