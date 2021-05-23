#!/bin/bash

#cmd="python ../run.py --experiment-name=$1 ${@:2} & python ../plot.py --experiment-name=$1 --repeat"
train_cmd="python ../run.py --experiment-name=$1 ${@:2}"
plot_cmd="python ../plot.py --experiment-name=$1; python ../plot.py --experiment-name=$1 --long"
cmd="( $train_cmd; $plot_cmd ) & ( sleep 900; $plot_cmd )"
echo $cmd
eval $cmd
