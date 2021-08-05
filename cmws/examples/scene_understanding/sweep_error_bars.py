import cmws.slurm_util
import cmws.examples.scene_understanding.run


def get_run_argss():
    experiment_name = "error_bars"

    mode = "cube"
    num_grid_rows = 2
    num_grid_cols = 2
    shrink_factor = 0.01
    num_seeds = 20
    num_primitives = 5

    for seed in range(num_seeds):
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
        args.learn_blur = True
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
        args.learn_blur = True
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
