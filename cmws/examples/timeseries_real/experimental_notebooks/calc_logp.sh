#!/bin/bash

#cmd="python ../run.py --experiment-name=$1 ${@:2} & python ../plot.py --experiment-name=$1 --repeat"

experiment_name=$1
for path in save/$experiment_name/*_1; do
    echo $path;
    python ../calc_logp.py --checkpoint-path=$path/checkpoints/latest.pt
done