import cmws.slurm_util
import cmws.examples.scene_understanding.run


def get_run_argss():
    experiment_name = "cmws_vs_rws_learnColor"

    mode = "cube"

    for seed in range(10):
        for num_grid_rows, num_grid_cols in [[2, 2]]:

            if num_grid_rows == 3:
                num_primitives = 15 # more primitives for 3x3
                shrink_factors = [0.01]
            else:
                num_primitives = 5
                shrink_factors = [0.01]

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
                args.algorithm = "cmws_5"
                args.model_type = "scene_understanding"
                args.continue_training = True
                args.mode = mode
                args.shrink_factor=shrink_factor
                yield args

                args = cmws.examples.scene_understanding.run.get_args_parser().parse_args([])
                args.experiment_name = experiment_name
                args.num_grid_rows = num_grid_rows
                args.num_grid_cols = num_grid_cols
                args.seed = seed
                args.num_primitives = num_primitives
                args.num_particles = 50
                args.insomnia = 0.50
                args.algorithm = "vimco_2"
                args.model_type = "scene_understanding"
                args.continue_training = True
                args.lr = 1e-4
                args.mode = mode
                args.shrink_factor = shrink_factor
                yield args

                # reinforce
                args = cmws.examples.scene_understanding.run.get_args_parser().parse_args([])
                args.experiment_name = experiment_name
                args.num_grid_rows = num_grid_rows
                args.num_grid_cols = num_grid_cols
                args.seed = seed
                args.num_primitives = num_primitives
                args.num_particles = 50
                args.insomnia = 0.50
                args.algorithm = "reinforce"
                args.model_type = "scene_understanding"
                args.lr = 1e-4
                args.continue_training = True
                args.mode = mode
                args.shrink_factor = shrink_factor
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
