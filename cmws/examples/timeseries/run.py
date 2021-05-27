from pathlib import Path

from cmws import train, util
from cmws.examples.timeseries import util as timeseries_util


def main(args):
    # Cuda
    device = util.get_device()

    # Seed
    util.set_seed(args.seed)

    # Initialize models, optimizer, stats, data
    checkpoint_path = util.get_checkpoint_path(args.experiment_name, get_config_name(args))
    if not (args.continue_training and Path(checkpoint_path).exists()):
        util.logging.info("Training from scratch")
        model, optimizer, stats = timeseries_util.init(args, device, fast=args.fast)
    else:
        model, optimizer, stats, _ = timeseries_util.load_checkpoint(checkpoint_path, device)

    # Train
    train.train(model, optimizer, stats, args)


def get_config_name(args):
    return f"{args.algorithm}_{args.seed}"


def get_args_parser():
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--experiment-name", default="", help=" ")
    parser.add_argument(
        "--algorithm",
        default="rws",
        choices=["rws", "elbo", "vimco", "cmws", "cmws_2", "cmws_3", "cmws_4", "cmws_5", "vimco_2", "reinforce"],
        help=" ",
    )
    parser.add_argument("--seed", default=1, type=int, help=" ")
    parser.add_argument("--batch-size", default=5, type=int, help=" ")
    parser.add_argument("--num-particles", default=50, type=int, help=" ")
    parser.add_argument("--memory-size", default=10, type=int, help=" ")
    parser.add_argument("--num-proposals-mws", default=10, type=int, help=" ")
    parser.add_argument("--test-num-particles", default=100, type=int, help=" ")

    parser.add_argument('--include-symbols', default="PXLWRCp12345xabcdel!@#$%", type=str)
    # Data
    parser.add_argument("--full-training-data", action="store_true", help=" ")
    parser.add_argument("--synthetic-data", action="store_true", help=" ")

    # Model
    parser.add_argument("--learn-eps", action="store_true", help=" ")
    parser.add_argument("--learn-coarse", action="store_true", help=" ")
    parser.add_argument("--max-num-chars", default=10, type=int, help=" ")
    parser.add_argument("--generative-model-lstm-hidden-dim", default=128, type=int, help=" ")
    parser.add_argument("--guide-lstm-hidden-dim", default=128, type=int, help=" ")
    parser.add_argument(
        "--model-type", default="timeseries", choices=["timeseries",], help=" ",
    )
    parser.add_argument("--allow_repeat_factors", action="store_true", help=" ")

    # Optimization
    parser.add_argument("--continue-training", action="store_true", help=" ")
    parser.add_argument("--num-iterations", default=1000, type=int, help=" ")
    parser.add_argument("--num-sleep-pretraining-iterations", default=0, type=int, help=" ")
    parser.add_argument("--sleep-pretraining-batch-size", default=0, type=int, help=" ")
    parser.add_argument("--lr", default=1e-3, type=float, help=" ")
    parser.add_argument("--lr-sleep-pretraining", default=None, type=float, help=" ")
    parser.add_argument("--lr-guide-continuous", default=None, type=float, help=" ")
    parser.add_argument("--lr-guide-discrete", default=None, type=float, help=" ")
    parser.add_argument("--lr-prior-continuous", default=None, type=float, help=" ")
    parser.add_argument("--lr-prior-discrete", default=None, type=float, help=" ")
    parser.add_argument("--lr-likelihood", default=None, type=float, help=" ")

    parser.add_argument(
        "--insomnia",
        default=1.0,
        type=float,
        help="only applicable for RWS for pyro models - 1.0 means Wake-Wake, 0.0 means Wake-Sleep,"
        "otherwise it's inbetween",
    )
    parser.add_argument(
        "--continuous-guide-lr", default=1e-3, type=float, help=" ",
    )
    parser.add_argument("--log-interval", default=1, type=int, help=" ")
    parser.add_argument("--save-interval", default=10, type=int, help=" ")
    parser.add_argument("--test-interval", default=250, type=int, help=" ")
    parser.add_argument("--checkpoint-interval", default=1000, type=int, help=" ")

    parser.add_argument("--fast", action="store_true", help=" ")
    return parser


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
