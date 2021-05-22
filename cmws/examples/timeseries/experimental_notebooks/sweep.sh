for num_particles in 10 20; do
    #for memory_size in 5 10; do
    memory_size=10;
        for num_proposals_mws in 5 10; do
           #for lr_continuous_latents in 0.02 0.005; do
	   lr_continuous_latents=0.01;
                for include_symbols in "WRCPpXxLl" "WRCPp" "WRCP12345"; do
			experiment_name=expt1_particles${num_particles}_memory${memory_size}_proposals${num_proposals_mws}_lrc${lr_continuous_latents}_symbols${include_symbols}
			cmd="sbatch --partition=tenenbaum --time=3:00:00 --output=logs/$experiment_name.out ./run.sh $experiment_name
			    --continue-training
			    --algorithm=cmws_5
			    --num-particles=$num_particles
			    --full-training-data
			    --generative-model-lstm-hidden-dim=10
			    --guide-lstm-hidden-dim=10
			    --memory-size=$memory_size
			    --num-proposals-mws=$num_proposals_mws
			    --max-num-chars=7
			    --lr=0.001
			    --lr-continuous-latents=$lr_continuous_latents
			    --lr-sleep-pretraining=0.01
				--include-symbols=$include_symbols
				"
			echo $cmd
			eval $cmd
		done
	    #done
        done
    #done
done
