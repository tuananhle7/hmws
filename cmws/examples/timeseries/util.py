import collections
import itertools
import time
import math
from pathlib import Path

import cmws
import torch
import torch.nn as nn
import torch.nn.functional as F
from cmws.examples.timeseries.models import timeseries
from cmws.util import logging
import cmws.examples.timeseries.expression_prior_pretraining

base_kernel_chars = {"W", "R", "E", "C"}
char_to_long_char = {"W": "WN", "R": "SE", "E": "Per", "C": "C"}
char_to_num = {"W": 0, "R": 1, "E": 2, "C": 3, "*": 4, "+": 5, "(": 6, ")": 7}
num_to_char = dict([(v, k) for k, v in char_to_num.items()])
vocabulary_size = len(char_to_num)
gp_params_dim = 5
epsilon = 1e-4


def get_raw_expression(expression, device):
    """
    Args
        expression (str)
        device

    Returns [num_chars, vocabulary_size]
    """
    raw_expression = []
    for char in expression:
        raw_expression.append(torch.tensor(char_to_num[char], device=device).long())
    return torch.stack(raw_expression)


def get_expression(raw_expression):
    """
    Args
        raw_expression [num_chars]

    Returns string of length num_chars
    """
    expression = ""
    for raw_char in raw_expression:
        expression += num_to_char[raw_char.item()]
    return expression


def get_long_expression(expression):
    """
    Args
        expression (str)

    Returns (str)
    """
    long_expression = ""
    for char in expression:
        if char in base_kernel_chars:
            long_expression += char_to_long_char[char]
        elif char == "*":
            long_expression += " × "
        elif char == "+":
            long_expression += " + "
        else:
            long_expression += char
    return long_expression


def count_base_kernels(raw_expression):
    """How many base kernels are there in the raw_expression?

    Args
        raw_expression [num_chars]

    Returns int
    """
    result = 0
    for char in get_expression(raw_expression):
        result += int(char in base_kernel_chars)
    return result


class ParsingError(Exception):
    pass


class Kernel(nn.Module):
    """Kernel -> Kernel + Kernel | Kernel * Kernel | W | R | E | C
    Adapted from
    https://github.com/insperatum/wsvae/blob/7dee0708587e6a33b7328206ce5edd8262d568b6/gp.py#L12

    Args
        expression (str) consists of characters *, +, (, ) or
            W = WhiteNoise
            R = SquaredExponential
            E = Periodic
            C = Constant
        raw_params [num_base_kernels, gp_params_dim=5]
            raw_params[i, 0] raw_scale (WhiteNoise)
            raw_params[i, 1] raw_lengthscale (SE)
            raw_params[i, 2] raw_period (Per)
            raw_params[i, 3] raw_lengthscale (Per)
            raw_params[i, 4] raw_const (Constant)
    """

    def __init__(self, expression, raw_params):
        super().__init__()
        self.expression = expression
        self.index = 0

        self.raw_params = raw_params
        self.raw_params_index = 0

        self.value = self.getValue()

    @property
    def device(self):
        return self.raw_params.device

    def getValue(self):
        value = self.parseExpression()
        if self.hasNext():
            raise ParsingError(
                "Unexpected character found: '" + self.peek() + "' at index " + str(self.index)
            )
        return value

    def peek(self):
        return self.expression[self.index : self.index + 1]

    def hasNext(self):
        return self.index < len(self.expression)

    def parseExpression(self):
        return self.parseAddition()

    def parseAddition(self):
        values = [self.parseMultiplication()]
        while True:
            char = self.peek()
            if char == "+":
                self.index += 1
                values.append(self.parseMultiplication())
            else:
                break
        if len(values) == 1:
            return values[0]
        else:
            return {"op": "+", "values": values}

    def parseMultiplication(self):
        values = [self.parseParenthesis()]
        while True:
            char = self.peek()
            if char == "*":
                self.index += 1
                values.append(self.parseParenthesis())
            else:
                break
        if len(values) == 1:
            return values[0]
        else:
            return {"op": "*", "values": values}

    def parseParenthesis(self):
        char = self.peek()
        if char == "(":
            self.index += 1
            value = self.parseExpression()
            if self.peek() != ")":
                raise ParsingError("No closing parenthesis found at character " + str(self.index))
            self.index += 1
            return value
        else:
            return self.parseValue()

    def parseValue(self):
        char = self.peek()
        self.index += 1
        if self.raw_params_index < len(self.raw_params):
            param = self.raw_params[self.raw_params_index]
            self.raw_params_index += 1
        if char == "W":
            return {"op": "WhiteNoise", "scale_sq": F.softplus(param[0]) + epsilon}
        elif char == "R":
            return {"op": "RBF", "lengthscale_sq": F.softplus(param[1]) + epsilon}
        elif char == "E":
            return {
                "op": "ExpSinSq",
                "period": F.softplus(param[2]) + epsilon,
                "lengthscale_sq": F.softplus(param[3]) + epsilon,
            }
        if char == "C":
            return {"op": "Constant", "const": F.softplus(param[4]) + epsilon}
        else:
            raise ParsingError("Cannot parse char: " + str(char))

    def forward(self, x_1, x_2, kernel=None):
        """
        Args
            x_1 [batch_size, num_timesteps_1, 1]
            x_2 [batch_size, 1, num_timesteps_2]

        Returns [batch_size, num_timesteps_1, num_timesteps_2]
        """
        if kernel is None:
            kernel = self.value
        if kernel["op"] == "+":
            t = self.forward(x_1, x_2, kernel["values"][0])
            for v in kernel["values"][1:]:
                t = t + self.forward(x_1, x_2, v)
            return t
        elif kernel["op"] == "*":
            t = self.forward(x_1, x_2, kernel["values"][0])
            for v in kernel["values"][1:]:
                t = t * self.forward(x_1, x_2, v)
            return t
        elif kernel["op"] == "Constant":
            batch_size, num_timesteps_1 = x_1.shape[:2]
            num_timesteps_2 = x_2.shape[-1]
            return (
                torch.ones((batch_size, num_timesteps_1, num_timesteps_2), device=x_1.device)
                * kernel["const"]
            )
        elif kernel["op"] == "WhiteNoise":
            return (x_1 == x_2).float() * kernel["scale_sq"]
        elif kernel["op"] == "RBF":
            l2 = kernel["lengthscale_sq"]
            return (-((x_1 - x_2) ** 2) / (2 * l2)).exp()
        elif kernel["op"] == "ExpSinSq":
            t = kernel["period"]
            l2 = kernel["lengthscale_sq"]
            return (-2 * (math.pi * (x_1 - x_2).abs() / t).sin() ** 2 / l2).exp()
        else:
            raise ParsingError(f"Cannot parse kernel op {kernel['op']}")


# Init, saving, etc
def init(run_args, device, fast=False):
    memory = None
    if run_args.model_type == "timeseries":
        # Generative model
        generative_model = timeseries.GenerativeModel(
            max_num_chars=run_args.max_num_chars, lstm_hidden_dim=run_args.lstm_hidden_dim
        ).to(device)

        # Guide
        guide = timeseries.Guide(
            max_num_chars=run_args.max_num_chars, lstm_hidden_dim=run_args.lstm_hidden_dim
        ).to(device)

        if not fast:
            # Pretrain the prior
            cmws.examples.timeseries.expression_prior_pretraining.pretrain_expression_prior(
                generative_model, batch_size=10, num_iterations=2000
            )

        # Memory
        if "mws" in run_args.algorithm:
            memory = cmws.memory.Memory(
                2000, run_args.memory_size, generative_model, check_unique=not fast
            ).to(device)

    # Model dict
    model = {"generative_model": generative_model, "guide": guide, "memory": memory}

    # Optimizer
    parameters = itertools.chain(generative_model.parameters(), guide.parameters())
    optimizer = torch.optim.Adam(parameters, lr=run_args.lr)

    # Stats
    stats = Stats([], [], [], [])

    return model, optimizer, stats


def save_checkpoint(path, model, optimizer, stats, run_args=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    generative_model, guide, memory = model["generative_model"], model["guide"], model["memory"]
    torch.save(
        {
            "generative_model_state_dict": generative_model.state_dict(),
            "guide_state_dict": guide.state_dict(),
            "memory_state_dict": None if memory is None else memory.state_dict(),
            "optimizer_state_dict": optimizer.get_state()
            if "_pyro" in run_args.model_type
            else optimizer.state_dict(),
            "stats": stats,
            "run_args": run_args,
        },
        path,
    )
    logging.info(f"Saved checkpoint to {path}")


def load_checkpoint(path, device, num_tries=3):
    for i in range(num_tries):
        try:
            checkpoint = torch.load(path, map_location=device)
            break
        except Exception as e:
            logging.info(f"Error {e}")
            wait_time = 2 ** i
            logging.info(f"Waiting for {wait_time} seconds")
            time.sleep(wait_time)
    run_args = checkpoint["run_args"]
    model, optimizer, stats = init(run_args, device, fast=True)

    generative_model, guide, memory = model["generative_model"], model["guide"], model["memory"]
    guide.load_state_dict(checkpoint["guide_state_dict"])
    generative_model.load_state_dict(checkpoint["generative_model_state_dict"])
    if memory is not None:
        memory.load_state_dict(checkpoint["memory_state_dict"])

    model = {"generative_model": generative_model, "guide": guide, "memory": memory}
    if "_pyro" in run_args.model_type:
        optimizer.set_state(checkpoint["optimizer_state_dict"])
    else:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    stats = checkpoint["stats"]
    return model, optimizer, stats, run_args


Stats = collections.namedtuple("Stats", ["losses", "sleep_pretraining_losses", "log_ps", "kls"])
