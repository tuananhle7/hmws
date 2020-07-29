import util
import argparse
import torch
import train


def main(args):
    # general
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        args.cuda = True
    else:
        device = torch.device("cpu")
        args.cuda = False
    util.set_seed(args.seed)

    generative_model, guide, optimizer, memory, stats = util.init(args, device)

    train.train(args, generative_model, guide, memory, optimizer, stats)


def get_args_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # general
    parser.add_argument("--cuda", action="store_true", help="use cuda")
    parser.add_argument("--seed", type=int, default=0, help=" ")
    parser.add_argument("--support-size", type=int, default=5, help=" ")
    parser.add_argument("--memory-size", type=int, default=3, help=" ")
    parser.add_argument("--num-particles", type=int, default=100, help=" ")
    parser.add_argument("--num-cmws-mc-samples", type=int, default=100, help=" ")
    parser.add_argument("--num-cmws-iterations", type=int, default=10, help=" ")
    parser.add_argument("--num-iterations", type=int, default=10000, help=" ")
    parser.add_argument("--save-interval", type=int, default=100, help=" ")
    parser.add_argument("--log-interval", type=int, default=10, help=" ")
    parser.add_argument(
        "--algorithm",
        default="rws",
        choices=["rws", "elbo", "mws", "cmws"],
        help="Learning/inference algorithm to use",
    )
    parser.add_argument(
        "--cmws-estimator",
        default="is",
        choices=["is", "sgd"],
        help="Inner inference algorithm for cmws",
    )
    return parser


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
