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
from cmws.examples.timeseries import data
from cmws.util import logging
import cmws.examples.timeseries.expression_prior_pretraining

include_scale = False
if include_scale:
    base_kernel_chars = {"W", "R", "E", "L"}
    char_to_long_char = {"W": "WN", "R": "SE", "E": "Per", "L": "Lin"}
    char_to_num = {"+":0, "*":1, "W":2, "R":3, "E":4, "L":5}
    num_to_char = dict([(v, k) for k, v in char_to_num.items()])
    gp_params_dim = 8
else:
    base_kernel_chars = {"W", "R", "E", "L", "C"}
    char_to_long_char = {"W": "WN", "R": "SE", "E": "Per", "L": "Lin", "C": "const"}
    char_to_num = {"+":0, "*":1, "W":2, "R":3, "E":4, "L":5, "C":6}
    num_to_char = dict([(v, k) for k, v in char_to_num.items()])
    gp_params_dim = 5

vocabulary_size = len(char_to_num)
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


def get_long_expression_with_params(expression, params):
    """
    Args
        expression (str)
        params

    Returns (str)
    """
    long_expression = ""
    params_idx = 0
    for char in expression:
        if char in base_kernel_chars:
            param = params[params_idx]
            params_idx += 1
            if char == "W":
                long_expression += f"{char_to_long_char[char]}({param:.2f})"
            elif char == "R":
                long_expression += f"{char_to_long_char[char]}({param[0]:.2f}, {param[1]:.2f})"
            elif char == "E":
                long_expression += (
                    f"{char_to_long_char[char]}({param[0]:.2f}, {param[1]:.2f}, {param[2]:.2f})"
                )
            elif char == "L":
                long_expression += f"{char_to_long_char[char]}({param[0]:.2f}, {param[1]:.2f})"
            elif char == "C":
                long_expression += f"{char_to_long_char[char]}({param:.2f})"
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
    """Kernel -> Kernel + Kernel | Kernel * Kernel | W | R | E | L
    Adapted from
    https://github.com/insperatum/wsvae/blob/7dee0708587e6a33b7328206ce5edd8262d568b6/gp.py#L12

    Args
        expression (str) consists of characters *, +, (, ) or
            W = WhiteNoise
            R = SquaredExponential
            E = Periodic
            C = Linear
        raw_params [num_base_kernels, gp_params_dim=5]
            raw_params[i, 0] raw_scale_sq (WhiteNoise)
            raw_params[i, 1] raw_scale_sq (SE)
            raw_params[i, 2] raw_lengthscale_sq (SE)
            raw_params[i, 3] raw_scale_sq (Per)
            raw_params[i, 4] raw_period (Per)
            raw_params[i, 5] raw_lengthscale_sq (Per)
            raw_params[i, 6] raw_scale_sq (Linear)
            raw_params[i, 7] raw_offset (Linear)
    """

    def __init__(self, expression, raw_params):
        super().__init__()
        self.expression = expression
        self.index = 0

        self.raw_params = raw_params
        self.raw_params_index = 0
        self.params = []

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
            raw_param = self.raw_params[self.raw_params_index]
            self.raw_params_index += 1

        if include_scale: 
            if char == "W":
                scale_sq = F.softplus(raw_param[0])*0.1
                self.params.append(scale_sq.item())
                return {"op": "WhiteNoise", "scale_sq": scale_sq}
            elif char == "R":
                scale_sq = F.softplus(raw_param[1])*0.1
                lengthscale_sq = F.softplus(raw_param[2])
                self.params.append([scale_sq.item(), lengthscale_sq.item()])
                return {"op": "RBF", "scale_sq": scale_sq, "lengthscale_sq": lengthscale_sq}
            elif char == "E":
                scale_sq = F.softplus(raw_param[3])*0.1
                period = torch.sigmoid(raw_param[3])
                lengthscale_sq = F.softplus(raw_param[4])
                self.params.append([scale_sq.item(), period.item(), lengthscale_sq.item()])
                return {
                    "op": "ExpSinSq",
                    "scale_sq": scale_sq,
                    "period": period,
                    "lengthscale_sq": lengthscale_sq,
                }
            elif char == "L":
                scale_sq = F.softplus(raw_param[6])*0.1
                offset = raw_param[7] + 1
                self.params.append([scale_sq.item(), offset.item()])
                return {"op": "Linear",  "scale_sq": scale_sq, "offset": offset}
        else:
            if char == "W":
                return {"op": "WhiteNoise"}
            elif char == "R":
                lengthscale_sq = F.softplus(raw_param[0])
                self.params.append(lengthscale_sq.item())
                return {"op": "RBF", "lengthscale_sq": lengthscale_sq}
            elif char == "E":
                period = torch.sigmoid(raw_param[1])
                lengthscale_sq = F.softplus(raw_param[2])
                self.params.append([period.item(), lengthscale_sq.item()])
                return {"op": "ExpSinSq", "period": period, "lengthscale_sq": lengthscale_sq, }
            elif char == "L":
                offset = raw_param[3] + 1
                self.params.append(offset.item())
                return {"op": "Linear",  "offset": offset}  
            elif char == "C":
                value = F.softplus(raw_param[4]) * 0.1
                self.params.append([value.item()])
                return {"op": "Constant", "value": value}

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
            result = t
        elif kernel["op"] == "*":
            t = self.forward(x_1, x_2, kernel["values"][0])
            for v in kernel["values"][1:]:
                t = t * self.forward(x_1, x_2, v)
            result = t
        elif kernel["op"] == "Linear":
            c = kernel["offset"]
            s2 = kernel.get("scale_sq", 1)
            result = (x_1 - c) * (x_2 - c) * s2
            # print(kernel['op'], "c",c, "s2", s2)
        elif kernel["op"] == "WhiteNoise":
            s2 = kernel.get("scale_sq", 1)
            result = (x_1 == x_2).float() * s2
            # print(kernel['op'], "s2", s2)
        elif kernel["op"] == "RBF":
            l2 = kernel["lengthscale_sq"]
            s2 = kernel.get("scale_sq", 1)
            result = (-((x_1 - x_2) ** 2) / (2 * l2)).exp() * s2
            # print(kernel['op'], "s2", l2, "s2", s2)
        elif kernel["op"] == "ExpSinSq":
            t = kernel["period"]
            l2 = kernel["lengthscale_sq"]
            s2 = kernel.get("scale_sq", 1)
            result = (-2 * (math.pi * (x_1 - x_2).abs() / t).sin() ** 2 / l2).exp() * s2
            # print(kernel['op'], "t", t, "l2", l2, "s2", s2)
        elif kernel["op"] == "Constant":
            c = kernel["value"]
            result = torch.full((x_1-x_2).shape, c, device=x_1.device)
        else:
            raise ParsingError(f"Cannot parse kernel op {kernel['op']}")

        # Test nan
        if torch.isnan(result).any():
            raise RuntimeError("Covariance nan")

        return result


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
                generative_model, guide, batch_size=10, num_iterations=2000
            )

        # Memory
        if "mws" in run_args.algorithm:
            memory = cmws.memory.Memory(
                len(
                    data.TimeseriesDataset(
                        device, test=False, full_data=run_args.full_training_data
                    )
                ),
                run_args.memory_size,
                generative_model,
                check_unique=not fast,
            ).to(device)

    # Model dict
    model = {"generative_model": generative_model, "guide": guide, "memory": memory}

    # Optimizer
    continuous_latent_parameters = list(guide.gp_params_extractor.parameters())
    other_parameters = [
            *generative_model.parameters(),
            *(set(guide.parameters()) - set(continuous_latent_parameters))
        ]

    optimizer = torch.optim.Adam([
        {'params': continuous_latent_parameters, 'lr':run_args.lr_continuous_latents},
        {'params': other_parameters},
    ], lr=run_args.lr)

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
