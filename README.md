# Continuous Memoised Wake-Sleep

## Train

For a single run, run
```
python run.py
```

To run a sweep, as specified by `sweep.get_sweep_argss`, run
```
python sweep.py
```

Both of these save checkpoints and logs into `save/<path_base>/` where each `path_base` corresponds to one run and is defined by `util.get_path_base_from_args`.

## Plot

To plot results, run
```
python plot.py
```
which will loop through folders in `save/` and plot.

To plot an the training progress animation, run
```
python plot_animation.py
```
which will save a gif in `save/animation/training_progress.gif`.