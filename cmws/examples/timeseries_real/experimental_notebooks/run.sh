#!/bin/bash

#cmd="python ../run.py --experiment-name=$1 ${@:2} & python ../plot.py --experiment-name=$1 --repeat"
experiment_name=$1
algorithm=$2
seed=$3
train_cmd="python ../run.py --experiment-name=$experiment_name --algorithm=$algorithm --seed=$seed ${@:4}"
eval_cmd="python ../calc_logp.py --checkpoint-path=./save/${experiment_name}/${algorithm}_${seed}/checkpoints/latest.pt --cpu"
plot_cmd="python ../plot.py --experiment-name=$1 --cpu; python ../plot.py --experiment-name=${experiment_name} --long --cpu"
# cmd="( $train_cmd; $plot_cmd; $eval_cmd ) & ( python ../plot.py --experiment-name=$1 --cpu --repeat )"
cmd="$train_cmd; $plot_cmd; $eval_cmd"
echo $cmd
eval $cmd
