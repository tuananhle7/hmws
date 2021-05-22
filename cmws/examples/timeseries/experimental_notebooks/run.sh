#!/bin/bash

cmd="python ../run.py --experiment-name=$1 ${@:2} & python ../plot.py --experiment-name=$1 --repeat"
echo $cmd
eval $cmd
