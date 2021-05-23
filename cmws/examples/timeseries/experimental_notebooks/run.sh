#!/bin/bash

#cmd="python ../run.py --experiment-name=$1 ${@:2} & python ../plot.py --experiment-name=$1 --repeat"
<<<<<<< HEAD
train_cmd="python ../run.py --experiment-name=$1 ${@:2}"
plot_cmd="python ../plot.py --experiment-name=$1; python ../plot.py --experiment-name=$1 --long"
cmd="( $train_cmd; $plot_cmd ) & ( sleep 900; $plot_cmd )"
=======
experiment_name=$1
algorithm=$2
train_cmd="python ../run.py --experiment-name=$experiment_name --algorithm=$algorithm ${@:3}"
eval_cmd="python ../calc_logp.py --checkpoint-path=./save/${experiment_name}/${algorithm}_1/checkpoints/latest.pt"
plot_cmd="python ../plot.py --experiment-name=$1; python ../plot.py --experiment-name=${experiment_name} --long"
cmd="( $train_cmd; $plot_cmd; $eval_cmd ) & ( sleep 900; $plot_cmd )"
>>>>>>> 56448e172cf8ebe810f695dd8bedfcfb28e74138
echo $cmd
eval $cmd
