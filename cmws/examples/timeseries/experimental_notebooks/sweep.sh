sbatch_args="--partition=tenenbaum --time=3:00:00 --mem=8G -c2"
sbatch_plot_args="--partition=tenenbaum --time=3:00:00 --mem=16G -c2"

num_iterations=2000
test_interval=100

for max_num_chars in 9; do
for hidden in 10; do
for num_particles in 3 8; do
for memory_size in 2; do
for num_proposals_mws in 3 8; do
lr_guide_continuous=0.01
for lr_guide_discrete in 0.01; do
for lr_prior_continuous in 0.0; do #lr_prior_continuous=0.0
lr_prior_discrete=0.0
lr_likelihood=0.01
for include_symbols in "WRCP1234567890Ll"; do
for learn_eps in "--learn-eps"; do
for allow_repeat_factors in ""; do #"--allow_repeat_factors" ""; do
for learn_coarse in "--learn-coarse"; do
for insomnia in 0.5; do
    experiment_name=expt15_K${num_particles}_M${memory_size}_N${num_proposals_mws}_h${hidden}_c${max_num_chars}_symbols${include_symbols}_ins${insomnia}_lrpc=${lr_prior_continuous}_lrgd=${lr_guide_discrete}$learn_eps$allow_repeat_factors$learn_coarse
    for seed in 1 2 3 4 5 6 7 8 9 10; do
	algorithm=cmws_5
	cmd="sbatch $sbatch_args --output=logs/${algorithm}_${experiment_name}_seed${seed}.out ./run.sh $experiment_name $algorithm $seed
	    --continue-training
	    --num-iterations=$num_iterations
            --test-interval=$test_interval
	    --num-particles=$num_particles
	    --full-training-data
	    --generative-model-lstm-hidden-dim=$hidden
	    --guide-lstm-hidden-dim=$hidden
	    --memory-size=$memory_size
	    --num-proposals-mws=$num_proposals_mws
	    --max-num-chars=$max_num_chars
	    --lr-guide-continuous=$lr_guide_continuous
	    --lr-guide-discrete=$lr_guide_discrete
	    --lr-prior-continuous=$lr_prior_continuous
	    --lr-prior-discrete=$lr_prior_discrete
	    --lr-likelihood=$lr_likelihood
	    --lr-sleep-pretraining=0.01
	    --num-sleep-pretraining-iterations=1000
	    --include-symbols=$include_symbols
            --insomnia=$insomnia
	    $learn_eps
	    $allow_repeat_factors
            $learn_coarse
	    "
	echo $cmd
	eval $cmd
	total_num_particles=`expr $num_particles \* \( $memory_size + $num_proposals_mws \)`
	algorithm=rws
	cmd="sbatch $sbatch_args --output=logs/${algorithm}_${experiment_name}_seed${seed}.out ./run.sh $experiment_name $algorithm $seed
	    --continue-training
	    --num-iterations=$num_iterations
            --test-interval=$test_interval
	    --num-particles=$total_num_particles
	    --full-training-data
	    --generative-model-lstm-hidden-dim=$hidden
	    --guide-lstm-hidden-dim=$hidden
	    --max-num-chars=$max_num_chars
	    --lr-guide-continuous=$lr_guide_continuous
	    --lr-guide-discrete=$lr_guide_discrete
	    --lr-prior-continuous=$lr_prior_continuous
	    --lr-prior-discrete=$lr_prior_discrete
	    --lr-likelihood=$lr_likelihood
	    --lr-sleep-pretraining=0.01
	    --num-sleep-pretraining-iterations=1000
	    --include-symbols=$include_symbols
            --insomnia=$insomnia
	    $learn_eps
	    $allow_repeat_factors
            $learn_coarse
	    "
	echo $cmd
	eval $cmd

    done
    cmd="sbatch $sbatch_plot_args --output=logs/plot_$experiment_name.out ./plot.sh $experiment_name"	
    echo $cmd
    eval $cmd

done
done
done
done
done
done
done
done
done
done
done
done
