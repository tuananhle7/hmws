import cmws.slurm_util
from cmws.examples.stacking import run


def get_run_argss():
    for seed in range(5):
        # CMWS
        args = run.get_args_parser().parse_args([])
        args.seed = seed
        args.num_particles = 10
        args.insomnia = 0.75
        args.algorithm = "cmws"
        args.model_type = "stacking"
        args.continue_training = True
        yield args

        # RWS
        args = run.get_args_parser().parse_args([])
        args.seed = seed
        args.num_particles = 50
        args.insomnia = 0.75
        args.algorithm = "rws"
        args.model_type = "stacking"
        args.continue_training = True
        yield args


def main(args):
    cmws.slurm_util.submit_slurm_jobs(list(get_run_argss()), args.no_repeat, args.cancel, args.rm)


if __name__ == "__main__":
    parser = cmws.slurm_util.get_parser()
    args = parser.parse_args()
    main(args)
