#!/bin/bash

#cmd="python ../run.py --experiment-name=$1 ${@:2} & python ../plot.py --experiment-name=$1 --repeat"
cmd="python ../run.py --experiment-name=$1 ${@:2} ; python ../plot.py --experiment-name=$1; python ../plot.py --experiment-name=$1 --long"
echo $cmd
eval $cmd
