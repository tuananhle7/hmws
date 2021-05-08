import cmws.slurm_util
from cmws.examples.timeseries import run


def get_run_argss():
    experiment_name = "2021_05_08_cmws_rws_seeds"

    for seed in range(5):
        # CMWS
        args = run.get_args_parser().parse_args([])
        args.experiment_name = experiment_name
        args.seed = seed
        args.full_training_data = True
        args.num_particles = 10
        args.insomnia = 1.0
        args.num_sleep_pretraining_iterations = 10000
        args.sleep_pretraining_batch_size = 50
        args.algorithm = "cmws"
        args.continue_training = True
        yield args

        # RWS
        args = run.get_args_parser().parse_args([])
        args.experiment_name = experiment_name
        args.seed = seed
        args.full_training_data = True
        args.num_particles = 50
        args.insomnia = 1.0
        args.num_sleep_pretraining_iterations = 10000
        args.sleep_pretraining_batch_size = 50
        args.algorithm = "rws"
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
