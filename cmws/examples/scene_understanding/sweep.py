import cmws.slurm_util
import cmws.examples.scene_understanding.run


def get_run_argss():
    experiment_name = "cmws_vs_rws_learnColor"
    for seed in range(10):
        for num_grid_rows, num_grid_cols in [[2, 2], [3, 3]]:

            if num_grid_rows == 3:
                num_primitives = 10 # increase primitives
                if seed >= 5: continue # only run 5 seeds for 3x3 for now
            else: num_primitives = 5

            # CMWS
            args = cmws.examples.scene_understanding.run.get_args_parser().parse_args([])
            args.experiment_name = experiment_name
            args.num_grid_rows = num_grid_rows
            args.num_grid_cols = num_grid_cols
            args.seed = seed
            args.num_primitives = num_primitives
            args.num_particles = 10
            args.memory_size = 3
            args.num_proposals_mws = 2
            args.insomnia = 0.50
            args.algorithm = "cmws_2"
            args.model_type = "scene_understanding"
            args.continue_training = True
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
