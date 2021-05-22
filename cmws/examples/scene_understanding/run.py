from pathlib import Path

from cmws import train, util
from cmws.examples.scene_understanding import util as scene_understanding_util


def main(args):
    # Cuda
    device = util.get_device()

    # Seed
    util.set_seed(args.seed)

    print("mode: ", args.mode)

    # Initialize models, optimizer, stats, data
    checkpoint_path = util.get_checkpoint_path(args.experiment_name, get_config_name(args))
    if not (args.continue_training and Path(checkpoint_path).exists()):
        util.logging.info("Training from scratch")
        model, optimizer, stats = scene_understanding_util.init(args, device)
    else:
        model, optimizer, stats, _ = scene_understanding_util.load_checkpoint(
            checkpoint_path, device
        )

    # Train
    train.train(model, optimizer, stats, args)


def get_config_name(args):
    return f"{args.algorithm}_{args.num_grid_rows}_{args.seed}"


def get_args_parser():
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--experiment-name", default="", help=" ")
    parser.add_argument(
        "--algorithm", default="rws", choices=["rws", "elbo", "vimco", "cmws", "cmws_2", "cmws_4"], help=" ",
    )
    parser.add_argument("--seed", default=1, type=int, help=" ")
    parser.add_argument("--batch-size", default=5, type=int, help=" ")
    parser.add_argument("--num-particles", default=50, type=int, help=" ")
    parser.add_argument("--memory-size", default=10, type=int, help=" ")
    parser.add_argument("--num-proposals-mws", default=10, type=int, help=" ")
    parser.add_argument("--test-num-particles", default=10, type=int, help=" ")
    parser.add_argument(
        "--mode", default="cube", choices=["block", "cube"],
        help="which primitives populate scene",
    )

    # Model
    parser.add_argument("--num-primitives", default=5, type=int, help=" ")
    parser.add_argument("--num-grid-rows", default=2, type=int, help=" ")
    parser.add_argument("--num-grid-cols", default=2, type=int, help=" ")
    parser.add_argument("--max-num-blocks", default=3, type=int, help=" ")
    parser.add_argument(
        "--model-type", default="scene_understanding", choices=["scene_understanding"], help=" ",
    )
    parser.add_argument("--remove-color", default=0, type=int, # not a boolean just to be sure don't encounter bool => string issues
                        help="Control whether color is rendered. Either 0 (don't remove color) or 1 (remove color).")

    # Data
    parser.add_argument("--data-num-primitives", default=3, type=int, help=" ")

    # Optimization
    parser.add_argument("--continue-training", action="store_true", help=" ")
    parser.add_argument("--num-iterations", default=100000, type=int, help=" ")
    parser.add_argument("--num-sleep-pretraining-iterations", default=0, type=int, help=" ")
    parser.add_argument("--lr", default=1e-3, type=float, help=" ")
    parser.add_argument(
        "--insomnia",
        default=1.0,
        type=float,
        help="only applicable for RWS for pyro models - 1.0 means Wake-Wake, 0.0 means Wake-Sleep,"
        "otherwise it's inbetween",
    )
    parser.add_argument("--log-interval", default=1, type=int, help=" ")
    parser.add_argument("--save-interval", default=10, type=int, help=" ")
    parser.add_argument("--test-interval", default=100, type=int, help=" ")
    parser.add_argument("--checkpoint-interval", default=10001, type=int, help=" ")

    return parser


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
