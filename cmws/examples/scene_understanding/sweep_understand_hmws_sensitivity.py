import cmws.slurm_util
import cmws.examples.scene_understanding.run


def get_run_argss():
    experiment_name = "hmws_param_sensitivity"

    mode = "cube"
    num_grid_rows = 2
    num_grid_cols = 2
    shrink_factor = 0.01
    num_seeds = 5
    num_primitives = 5

    # hold either two of (hmws_num_particles, hmws_memory_size, hmw_num_proposals) fixed
    # let the other vary from the following:
    hmws_param_settings = [1,2,5,10]

    for seed in range(num_seeds):
        for param_val in hmws_param_settings:

            # vary num particles
            args = cmws.examples.scene_understanding.run.get_args_parser().parse_args([])
            args.experiment_name = experiment_name
            args.num_grid_rows = num_grid_rows
            args.num_grid_cols = num_grid_cols
            args.seed = seed
            args.num_primitives = num_primitives
            args.num_particles = param_val
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

            # vary memory size
            args = cmws.examples.scene_understanding.run.get_args_parser().parse_args([])
            args.experiment_name = experiment_name
            args.num_grid_rows = num_grid_rows
            args.num_grid_cols = num_grid_cols
            args.seed = seed
            args.num_primitives = num_primitives
            args.num_particles = 5
            args.memory_size = param_val
            args.num_proposals_mws = 5
            args.insomnia = 0.50
            args.algorithm = "cmws_5"
            args.model_type = "scene_understanding"
            args.continue_training = True
            args.mode = mode
            args.shrink_factor = shrink_factor
            args.learn_blur = True
            yield args

            # vary num proposals
            args = cmws.examples.scene_understanding.run.get_args_parser().parse_args([])
            args.experiment_name = experiment_name
            args.num_grid_rows = num_grid_rows
            args.num_grid_cols = num_grid_cols
            args.seed = seed
            args.num_primitives = num_primitives
            args.num_particles = 5
            args.memory_size = 5
            args.num_proposals_mws = param_val
            args.insomnia = 0.50
            args.algorithm = "cmws_5"
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
