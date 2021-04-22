import cmws.slurm_util
from cmws.examples.stacking_3d import run


def get_run_argss():
    for num_sleep_pretraining_iterations in [0, 1000]:
        for insomnia in [0.0, 0.25, 0.5, 0.75, 1.0]:
            args = run.get_args_parser().parse_args([])
            args.model_type = "stacking"
            args.num_sleep_pretraining_iterations = num_sleep_pretraining_iterations
            args.insomnia = insomnia
            args.num_primitives = 5
            args.max_num_blocks = 3

            # Basically don't test
            args.test_num_particles = 2
            args.test_interval = 10000
            args.continue_training = True
            yield args


def main(args):
    cmws.slurm_util.submit_slurm_jobs(list(get_run_argss()), args.no_repeat, args.cancel, args.rm)


if __name__ == "__main__":
    parser = cmws.slurm_util.get_parser()
    args = parser.parse_args()
    main(args)
