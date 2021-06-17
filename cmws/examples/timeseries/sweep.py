import cmws.slurm_util
from cmws.examples.timeseries import run


def get_run_argss():
    experiment_name = "2021_05_28_eval_test_new"

    for seed in range(5):
        # CMWS4
        args = run.get_args_parser().parse_args([])
        args.experiment_name = experiment_name
        args.continuous_guide_lr = 1e-3
        args.seed = seed
        args.full_training_data = True
        args.synthetic_data = True
        args.num_particles = 5
        args.memory_size = 5
        args.num_proposals_mws = 5
        args.insomnia = 0.5
        args.num_sleep_pretraining_iterations = 100
        args.sleep_pretraining_batch_size = 50
        args.algorithm = "cmws_5"
        args.continue_training = True
        args.num_iterations = 10000
        yield args

        # Baselines
        for algorithm in ["rws"]:
            args = run.get_args_parser().parse_args([])
            args.experiment_name = experiment_name
            args.seed = seed
            args.full_training_data = True
            args.synthetic_data = True
            args.num_particles = 50
            args.insomnia = 0.5
            args.num_sleep_pretraining_iterations = 100
            args.sleep_pretraining_batch_size = 50
            args.algorithm = algorithm
            args.continue_training = True
            args.num_iterations = 10000
            yield args

        for algorithm in ["vimco_2", "reinforce"]:
            args = run.get_args_parser().parse_args([])
            args.experiment_name = experiment_name
            args.lr = 5e-5
            args.seed = seed
            args.full_training_data = True
            args.synthetic_data = True
            args.num_particles = 50
            args.insomnia = 0.5
            args.num_sleep_pretraining_iterations = 100
            args.sleep_pretraining_batch_size = 50
            args.algorithm = algorithm
            args.continue_training = True
            args.num_iterations = 10000
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
