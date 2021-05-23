#!/bin/bash

#cmd="python ../run.py --experiment-name=$1 ${@:2} & python ../plot.py --experiment-name=$1 --repeat"
experiment_name=$1
cmd="python ../plot.py --experiment-name=$experiment_name --cpu --repeat"
echo $cmd
eval $cmd
