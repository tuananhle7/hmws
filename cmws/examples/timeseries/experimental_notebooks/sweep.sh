sbatch_args="--gres=gpu:1 --partition=tenenbaum --time=9:00:00 --mem=12G -c2"

for num_particles in 15; do
    for memory_size in 10; do
        for num_proposals_mws in 10; do
           for lr_continuous_latents in 0.01; do
                for include_symbols in "WRCPp" "WRCP12345" "WRCP12345Ll" "WRCP12345L!@#$%"; do

			experiment_name=expt3_particles${num_particles}_memory${memory_size}_proposals${num_proposals_mws}_lrc${lr_continuous_latents}_symbols${include_symbols}
			cmd="sbatch $sbatch_args --output=logs/cmws_4_$experiment_name.out ./run.sh $experiment_name
			    --continue-training
			    --algorithm=cmws_4
			    --num-particles=$num_particles
			    --full-training-data
			    --generative-model-lstm-hidden-dim=10
			    --guide-lstm-hidden-dim=10
			    --memory-size=$memory_size
			    --num-proposals-mws=$num_proposals_mws
			    --max-num-chars=7
			    --lr=0.0
			    --lr-continuous-latents=$lr_continuous_latents
			    --lr-sleep-pretraining=0.01
			    --num-sleep-pretraining-iterations=1000
			    --include-symbols=$include_symbols"
			echo $cmd
			eval $cmd

			total_num_particles=`expr $num_particles \* \( $memory_size + $num_proposals_mws \)`
			cmd="sbatch $sbatch_args --output=logs/rws_$experiment_name.out ./run.sh $experiment_name
			    --continue-training
			    --algorithm=rws
			    --num-particles=$total_num_particles
			    --full-training-data
			    --generative-model-lstm-hidden-dim=10
			    --guide-lstm-hidden-dim=10
			    --max-num-chars=7
			    --lr=0.0
			    --lr-continuous-latents=$lr_continuous_latents
			    --lr-sleep-pretraining=0.01
			    --num-sleep-pretraining-iterations=1000
			    --include-symbols=$include_symbols"
			echo $cmd
			eval $cmd
		done
	    done
        done
    done
done
