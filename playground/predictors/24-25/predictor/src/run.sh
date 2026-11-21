# Run the whole pipeline: data, base models, ensembles, scoring, predictions.
# Each step can also be run on its own (see README.md).
#
# Usage: bash run.sh
set -e
cd "$(dirname "$0")"

python prepare_data.py
bash folds.sh        # also writes the fold files (create_folds.py)
bash ensemble.sh
python deploy.py
python compare.py
python predictor.py
python performance.py
