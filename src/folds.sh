# Create the fold files, then train every model for every fold and position.
# Folds come from config.py (VALID_FOLDS, then TEST_FOLDS). Models come from model_dispatcher.
#
# Usage:
#   bash folds.sh                 # every model
#   bash folds.sh ridge cat       # only these models
set -e
cd "$(dirname "$0")"

POSITIONS=$(python -c "import config; print(*config.POSITIONS)")
FOLDS=$(python -c "import config; print(*config.VALID_FOLDS, *config.TEST_FOLDS)")
ALL_MODELS=$(python -c "import model_dispatcher as md; print(*md.models_gk)")
MODELS=${*:-$ALL_MODELS}

for model in $MODELS; do
  if [[ " $ALL_MODELS " != *" $model "* ]]; then
    echo "Unknown model '$model'. Available: $ALL_MODELS" >&2
    exit 1
  fi
done

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
  python create_folds.py --pos "$pos"
done

for model in $MODELS; do
  for fold in $FOLDS; do
    for pos in $POSITIONS; do
      run train.py --fold "$fold" --model "$model" --pos "$pos"
    done
  done
done
