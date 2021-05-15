import cmws.slurm_util
from cmws.examples.scene_understanding import run


def get_run_argss():
    experiment_name = "2021_05_13_cmws_vs_rws_new_data"
    for seed in range(1):
        for num_grid_rows, num_grid_cols in [[2, 2], [3, 3]]:
            # CMWS
            args = run.get_args_parser().parse_args([])
            args.experiment_name = experiment_name
            args.num_grid_rows = num_grid_rows
            args.num_grid_cols = num_grid_cols
            args.seed = seed
            args.num_particles = 10
            args.memory_size = 3
            args.num_proposals_mws = 2
            args.insomnia = 0.50
            args.algorithm = "cmws_2"
            args.model_type = "scene_understanding"
            args.continue_training = True
            yield args

            # RWS
            args = run.get_args_parser().parse_args([])
            args.experiment_name = experiment_name
            args.num_grid_rows = num_grid_rows
            args.num_grid_cols = num_grid_cols
            args.seed = seed
            args.num_particles = 50
            args.insomnia = 0.50
            args.algorithm = "rws"
            args.model_type = "scene_understanding"
            args.continue_training = True
            yield args


def get_job_name(run_args):
    return run.get_config_name(run_args)


def main(args):
    cmws.slurm_util.submit_slurm_jobs(
        get_run_argss, run.get_config_name, get_job_name, args.no_repeat, args.cancel, args.rm
    )


if __name__ == "__main__":
    parser = cmws.slurm_util.get_parser()
    args = parser.parse_args()
    main(args)
