import collections
import itertools
import time
import math
from pathlib import Path

import cmws
import torch
import torch.nn as nn
from cmws.examples.timeseries.models import timeseries
from cmws.examples.timeseries import data
from cmws.util import logging
import cmws.examples.timeseries.expression_prior_pretraining
from cmws.examples.timeseries import lstm_util

base_kernel_chars = {"W", "R", "1", "2", "3", "4", "C"}
char_to_long_char = {
    "W": "WN",
    "R": "SE",
    "1": "Per_Short",
    "2": "Per_Medium",
    "3": "Per_Long",
    "4": "Per_XLong",
    # "L": "Lin",
    "C": "Const",
}
char_to_num = {
    "W": 0,
    "R": 1,
    "1": 2,
    "2": 3,
    "3": 4,
    "4": 5,
    # "L": 6,
    "C": 6,
    "*": 7,
    "+": 8,
    "(": 9,
    ")": 10,
}
num_to_char = dict([(v, k) for k, v in char_to_num.items()])
vocabulary_size = len(char_to_num)
gp_params_dim = 16
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
            elif char == "1" or char == "2" or char == "3" or char == "4":
                long_expression += (
                    f"{char_to_long_char[char]}({param[0]:.2f}, {param[1]:.2f}, {param[2]:.2f})"
                )
            elif char == "L":
                long_expression += (
                    f"{char_to_long_char[char]}({param[0]:.2f}, {param[1]:.2f}, {param[2]:.2f})"
                )
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


def get_num_base_kernels(raw_expression, eos):
    """
    Args:
        raw_expression [*shape, max_num_chars]
        eos [*shape, max_num_chars]

    Returns: [*shape]
    """
    # Extract
    device = raw_expression.device
    max_num_chars = raw_expression.shape[-1]
    shape = raw_expression.shape[:-1]
    num_elements = cmws.util.get_num_elements(shape)

    # Flatten
    raw_expression_flattened = raw_expression.view(-1, max_num_chars)
    eos_flattened = eos.view(-1, max_num_chars)

    # Compute num timesteps
    # [num_elements]
    num_timesteps_flattened = lstm_util.get_num_timesteps(eos_flattened)

    result = []
    for element_id in range(num_elements):
        result.append(
            count_base_kernels(
                raw_expression_flattened[element_id, : num_timesteps_flattened[element_id]]
            )
        )
    return torch.tensor(result, device=device).long().view(shape)


def get_full_expression(raw_expression, eos, raw_gp_params, param_range=0.02):
    num_chars = lstm_util.get_num_timesteps(eos)
    num_base_kernels = get_num_base_kernels(raw_expression, eos)
    long_expression = get_long_expression(get_expression(raw_expression[:num_chars]))
    try:
        kernel = Kernel(
            get_expression(raw_expression[:num_chars]),
            raw_gp_params[:num_base_kernels],
            param_range=param_range,
        )
        return get_long_expression_with_params(
            get_expression(raw_expression[:num_chars]), kernel.params
        )
    except ParsingError as e:
        print(e)
        return long_expression


class ParsingError(Exception):
    pass


def get_interval(center, range_):
    return center - range_ / 2, center + range_ / 2


class Kernel(nn.Module):
    """Kernel -> Kernel + Kernel | Kernel * Kernel | W | R | E | L | C
    Adapted from
    https://github.com/insperatum/wsvae/blob/7dee0708587e6a33b7328206ce5edd8262d568b6/gp.py#L12

    Args
        expression (str) consists of characters *, +, (, ) or
            W = WhiteNoise
            R = SquaredExponential
            1 = Periodic Short
            2 = Periodic Medium
            3 = Periodic Long
            4 = Periodic Extra Long
            L = Linear
            C = Constant
        raw_params [num_base_kernels, gp_params_dim=14]
            raw_params[i, 0] raw_scale_sq (WhiteNoise)
            raw_params[i, 1] raw_scale_sq (SE)
            raw_params[i, 2] raw_lengthscale_sq (SE)
            raw_params[i, 3] raw_scale_sq (Per_S)
            raw_params[i, 4] raw_period (Per_S)
            raw_params[i, 5] raw_lengthscale_sq (Per_S)
            raw_params[i, 6] raw_scale_sq (Per_M)
            raw_params[i, 7] raw_period (Per_M)
            raw_params[i, 8] raw_lengthscale_sq (Per_M)
            raw_params[i, 9] raw_scale_sq (Per_L)
            raw_params[i, 10] raw_period (Per_L)
            raw_params[i, 11] raw_lengthscale_sq (Per_L)
            raw_params[i, 12] raw_scale_sq (Per_XL)
            raw_params[i, 13] raw_period (Per_XL)
            raw_params[i, 14] raw_lengthscale_sq (Per_XL)
            # raw_params[i, 15] raw_scale_b_sq (Linear)
            # raw_params[i, 16] raw_scale_sq (Linear)
            # raw_params[i, 17] raw_offset (Linear)
            raw_params[i, 15] raw_const (Constant)
    """

    def __init__(self, expression, raw_params, param_range=0.02):
        super().__init__()
        self.expression = expression
        self.index = 0

        self.raw_params = raw_params
        self.raw_params_index = 0
        self.params = []
        self.param_range = param_range

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
        if char == "W":
            # scale_sq = F.softplus(raw_param[0]) + 1e-2

            # scale_sq = torch.tensor(0.5, device=self.device)

            min_val, max_val = get_interval(0.5, self.param_range)
            scale_sq = torch.sigmoid(raw_param[0]) * (max_val - min_val) + min_val
            self.params.append(scale_sq.item())
            return {"op": "WhiteNoise", "scale_sq": scale_sq}
        elif char == "R":
            # scale_sq = F.softplus(raw_param[1]) + 1e-2
            # lengthscale_sq = F.softplus(raw_param[2]) + 0.1

            # scale_sq = torch.tensor(0.5, device=self.device)
            # lengthscale_sq = torch.tensor(0.5, device=self.device)

            min_val, max_val = get_interval(0.5, self.param_range)
            scale_sq = torch.sigmoid(raw_param[1]) * (max_val - min_val) + min_val
            lengthscale_sq = torch.sigmoid(raw_param[2]) * (max_val - min_val) + min_val
            self.params.append([scale_sq.item(), lengthscale_sq.item()])
            return {"op": "RBF", "scale_sq": scale_sq, "lengthscale_sq": lengthscale_sq}
        elif char == "1":
            # scale_sq = F.softplus(raw_param[3]) + 1e-2
            # min_period, max_period = 0.1, 0.2
            # period = torch.sigmoid(raw_param[4]) * (max_period - min_period) + min_period
            # lengthscale_sq = F.softplus(raw_param[5]) + 1e-1

            # scale_sq = torch.tensor(0.5, device=self.device)
            # period = torch.tensor(0.2, device=self.device)
            # lengthscale_sq = torch.tensor(0.5, device=self.device)

            min_val, max_val = get_interval(0.5, self.param_range)
            scale_sq = torch.sigmoid(raw_param[3]) * (max_val - min_val) + min_val
            min_val, max_val = get_interval(0.2, self.param_range)
            period = torch.sigmoid(raw_param[4]) * (max_val - min_val) + min_val
            min_val, max_val = get_interval(0.5, self.param_range)
            lengthscale_sq = torch.sigmoid(raw_param[5]) * (max_val - min_val) + min_val

            self.params.append([scale_sq.item(), period.item(), lengthscale_sq.item()])
            return {
                "op": "ExpSinSq",
                "scale_sq": scale_sq,
                "period": period,
                "lengthscale_sq": lengthscale_sq,
            }
        elif char == "2":
            # scale_sq = F.softplus(raw_param[6]) + 1e-2
            # min_period, max_period = 0.2, 0.5
            # period = torch.sigmoid(raw_param[7]) * (max_period - min_period) + min_period
            # lengthscale_sq = F.softplus(raw_param[8]) + 1e-1

            # scale_sq = torch.tensor(0.5, device=self.device)
            # period = torch.tensor(0.5, device=self.device)
            # lengthscale_sq = torch.tensor(0.5, device=self.device)

            min_val, max_val = get_interval(0.5, self.param_range)
            scale_sq = torch.sigmoid(raw_param[6]) * (max_val - min_val) + min_val
            min_val, max_val = get_interval(0.5, self.param_range)
            period = torch.sigmoid(raw_param[7]) * (max_val - min_val) + min_val
            min_val, max_val = get_interval(0.5, self.param_range)
            lengthscale_sq = torch.sigmoid(raw_param[8]) * (max_val - min_val) + min_val

            self.params.append([scale_sq.item(), period.item(), lengthscale_sq.item()])
            return {
                "op": "ExpSinSq",
                "scale_sq": scale_sq,
                "period": period,
                "lengthscale_sq": lengthscale_sq,
            }
        elif char == "3":
            # scale_sq = F.softplus(raw_param[9]) + 1e-2
            # min_period, max_period = 0.5, 1.0
            # period = torch.sigmoid(raw_param[10]) * (max_period - min_period) + min_period
            # lengthscale_sq = F.softplus(raw_param[11]) + 1e-1

            # scale_sq = torch.tensor(0.5, device=self.device)
            # period = torch.tensor(1.0, device=self.device)
            # lengthscale_sq = torch.tensor(0.5, device=self.device)

            min_val, max_val = get_interval(0.5, self.param_range)
            scale_sq = torch.sigmoid(raw_param[9]) * (max_val - min_val) + min_val
            min_val, max_val = get_interval(1.0, self.param_range)
            period = torch.sigmoid(raw_param[10]) * (max_val - min_val) + min_val
            min_val, max_val = get_interval(0.5, self.param_range)
            lengthscale_sq = torch.sigmoid(raw_param[11]) * (max_val - min_val) + min_val

            self.params.append([scale_sq.item(), period.item(), lengthscale_sq.item()])
            return {
                "op": "ExpSinSq",
                "scale_sq": scale_sq,
                "period": period,
                "lengthscale_sq": lengthscale_sq,
            }
        elif char == "4":
            # scale_sq = F.softplus(raw_param[12]) + 1e-2
            # # period = F.softplus(raw_param[13]) + 1.0
            # min_period, max_period = 1.0, 4.0
            # period = torch.sigmoid(raw_param[13]) * (max_period - min_period) + min_period
            # lengthscale_sq = F.softplus(raw_param[14]) + 1e-1

            # scale_sq = torch.tensor(0.5, device=self.device)
            # period = torch.tensor(1.5, device=self.device)
            # lengthscale_sq = torch.tensor(0.5, device=self.device)

            min_val, max_val = get_interval(0.5, self.param_range)
            scale_sq = torch.sigmoid(raw_param[12]) * (max_val - min_val) + min_val
            min_val, max_val = get_interval(1.5, self.param_range)
            period = torch.sigmoid(raw_param[13]) * (max_val - min_val) + min_val
            min_val, max_val = get_interval(0.5, self.param_range)
            lengthscale_sq = torch.sigmoid(raw_param[14]) * (max_val - min_val) + min_val

            self.params.append([scale_sq.item(), period.item(), lengthscale_sq.item()])
            return {
                "op": "ExpSinSq",
                "scale_sq": scale_sq,
                "period": period,
                "lengthscale_sq": lengthscale_sq,
            }
        # elif char == "L":
        #     scale_b_sq = F.softplus(raw_param[15]) + 1e-1
        #     scale_sq = F.softplus(raw_param[16]) + 1e-4
        #     min_offset, max_offset = -1.5, 1.5
        #     offset = torch.sigmoid(raw_param[17]) * (max_offset - min_offset) + min_offset
        #     self.params.append([scale_b_sq.item(), scale_sq.item(), offset.item()])
        #     return {
        #         "op": "Linear",
        #         "scale_b_sq": scale_b_sq,
        #         "scale_sq": scale_sq,
        #         "offset": offset,
        #     }
        elif char == "C":
            # const = F.softplus(raw_param[15]) + 1e-2

            # const = torch.tensor(0.5, device=self.device)

            min_val, max_val = get_interval(0.5, self.param_range)
            const = torch.sigmoid(raw_param[15]) * (max_val - min_val) + min_val

            self.params.append(const.item())
            return {"op": "Constant", "const": const}
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
            result = t
        elif kernel["op"] == "*":
            t = self.forward(x_1, x_2, kernel["values"][0])
            for v in kernel["values"][1:]:
                t = t * self.forward(x_1, x_2, v)
            result = t
        elif kernel["op"] == "Linear":
            sb2 = kernel["scale_b_sq"]
            c = kernel["offset"]
            s2 = kernel["scale_sq"]
            result = sb2 + (x_1 - c) * (x_2 - c) * s2
        elif kernel["op"] == "WhiteNoise":
            result = (x_1 == x_2).float() * kernel["scale_sq"]
        elif kernel["op"] == "RBF":
            l2 = kernel["lengthscale_sq"]
            s2 = kernel["scale_sq"]
            result = (-((x_1 - x_2) ** 2) / (2 * l2)).exp() * s2
        elif kernel["op"] == "ExpSinSq":
            t = kernel["period"]
            l2 = kernel["lengthscale_sq"]
            s2 = kernel["scale_sq"]
            result = (-2 * (math.pi * (x_1 - x_2).abs() / t).sin() ** 2 / l2).exp() * s2
        elif kernel["op"] == "Constant":
            # Extract
            batch_size, num_timesteps_1, _ = x_1.shape
            num_timesteps_2 = x_2.shape[-1]

            const = kernel["const"]
            result = (
                torch.ones((batch_size, num_timesteps_1, num_timesteps_2), device=self.device)
                * const
            )
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
            max_num_chars=run_args.max_num_chars,
            lstm_hidden_dim=run_args.lstm_hidden_dim,
            gp_param_range=run_args.continuous_param_range,
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
    guide_continuous_params = itertools.chain(
        guide.gp_params_lstm.parameters(), guide.gp_params_extractor.parameters()
    )
    guide_non_continuous_params = itertools.chain(
        guide.obs_embedder.parameters(),
        guide.expression_embedder.parameters(),
        guide.expression_lstm.parameters(),
        guide.expression_extractor.parameters(),
    )
    optimizer = torch.optim.Adam(
        [
            {"params": itertools.chain(generative_model.parameters(), guide_non_continuous_params)},
            {"params": guide_continuous_params, "lr": run_args.continuous_guide_lr},
        ],
        lr=run_args.lr,
    )
    # parameters = itertools.chain(generative_model.parameters(), guide.parameters())
    # optimizer = torch.optim.Adam(parameters, lr=run_args.lr)

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
