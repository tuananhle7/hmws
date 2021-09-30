import cmws.slurm_util
from cmws.examples.switching_ssm import run


def get_run_argss():
    experiment_name = "2021_09_30_slds_test_one_obs"

    for seed in range(10):
        # CMWS4
        args = run.get_args_parser().parse_args([])
        args.experiment_name = experiment_name
        args.seed = seed
        args.num_particles = 5
        args.memory_size = 5
        args.num_proposals_mws = 5
        args.insomnia = 0.5
        args.algorithm = "cmws_5"
        args.continue_training = True
        args.num_iterations = 20000
        args.batch_size = 1
        yield args

        # Baselines
        for algorithm in ["rws", "vimco_2", "reinforce"]:
            args = run.get_args_parser().parse_args([])
            args.experiment_name = experiment_name
            args.seed = seed
            args.num_particles = 50
            args.insomnia = 0.5
            args.algorithm = algorithm
            args.continue_training = True
            args.num_iterations = 20000
            args.batch_size = 1
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
