#!/bin/bash
# runs one sweep job; $1 = method, $2 = r (empty for base)
cd /home/ilyandho/Dev/FS-Go/thesis
name=$1${2:+_r$2}
python3 -u run_r_sweep_2425.py run $1 $2 > logs/sweep_$name.log 2>&1
rc=$?
echo "$(date +%H:%M) $name exit $rc" >> logs/sweep_status.log
