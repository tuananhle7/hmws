import cmws.slurm_util
import cmws.examples.scene_understanding.run


def get_run_argss():
    experiment_name = "cmws_vs_rws_learnColor"#"cmws_vs_rws_learnColor_shrink001"

    mode = "cube"

    for seed in range(5):#range(10):
        for num_grid_rows, num_grid_cols in [[2, 2], [3, 3]]:

            if num_grid_rows == 3:
                num_primitives = 20 # increase primitives
                #if seed >= 5: continue # only run 5 seeds for 3x3 for now
                shrink_factors = [0.01]
            else:
                num_primitives = 10
                shrink_factors = [0.01, 0.02, 0.05, 0.1, 0.2]

            for shrink_factor in shrink_factors:
                # CMWS
                args = cmws.examples.scene_understanding.run.get_args_parser().parse_args([])
                args.experiment_name = experiment_name
                args.num_grid_rows = num_grid_rows
                args.num_grid_cols = num_grid_cols
                args.seed = seed
                args.num_primitives = num_primitives
                args.num_particles = 5
                args.memory_size = 5
                args.num_proposals_mws = 5
                args.insomnia = 0.50
                args.algorithm = "cmws_4"
                args.model_type = "scene_understanding"
                args.continue_training = True
                args.mode = mode
                args.shrink_factor=shrink_factor
                yield args

                # RWS
                args = cmws.examples.scene_understanding.run.get_args_parser().parse_args([])
                args.experiment_name = experiment_name
                args.num_grid_rows = num_grid_rows
                args.num_grid_cols = num_grid_cols
                args.seed = seed
                args.num_primitives = num_primitives
                args.num_particles = 50
                args.insomnia = 0.50
                args.algorithm = "rws"
                args.model_type = "scene_understanding"
                args.continue_training = True
                args.mode = mode
                args.shrink_factor = shrink_factor
                yield args


def get_job_name(run_args):
    return cmws.examples.scene_understanding.run.get_config_name(run_args)


def main(args):
    cmws.slurm_util.submit_slurm_jobs(
        get_run_argss, cmws.examples.scene_understanding.run.get_config_name, get_job_name, args.no_repeat, args.cancel, args.rm
    )


if __name__ == "__main__":
    parser = cmws.slurm_util.get_parser()
    args = parser.parse_args()
    main(args)
