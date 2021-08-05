import cmws.slurm_util
import cmws.examples.scene_understanding.run


def get_run_argss():
    experiment_name = "num_particle_sensitivity"

    mode = "cube"
    num_grid_rows = 2
    num_grid_cols = 2
    shrink_factor = 0.01
    num_seeds = 5
    num_primitives = 5

    # form (hmws_num_particles, hmws_memory_size, hmw_num_proposals)
    # rws_num_particles = hmws_num_particles * (hmws_memory_size + hmw_num_proposals)
    hmws_param_settings = [(2,2,2), (3,3,3), (4,4,4), (5,5,5), (6,6,6), (7,7,7)]

    for seed in range(num_seeds):
        for (hmws_num_particles, hmws_memory_size, hmw_num_proposals) in hmws_param_settings:
            args = cmws.examples.scene_understanding.run.get_args_parser().parse_args([])
            args.experiment_name = experiment_name
            args.num_grid_rows = num_grid_rows
            args.num_grid_cols = num_grid_cols
            args.seed = seed
            args.num_primitives = num_primitives
            args.num_particles = hmws_num_particles
            args.memory_size = hmws_memory_size
            args.num_proposals_mws = hmw_num_proposals
            args.insomnia = 0.50
            args.algorithm = "cmws_5"
            args.model_type = "scene_understanding"
            args.continue_training = True
            args.mode = mode
            args.shrink_factor=shrink_factor
            args.learn_blur = True
            yield args

            # RWS
            rws_num_particles = hmws_num_particles * (hmws_memory_size + hmw_num_proposals)
            args = cmws.examples.scene_understanding.run.get_args_parser().parse_args([])
            args.experiment_name = experiment_name
            args.num_grid_rows = num_grid_rows
            args.num_grid_cols = num_grid_cols
            args.seed = seed
            args.num_primitives = num_primitives
            args.num_particles = rws_num_particles
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
