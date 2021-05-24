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
def init_symbols(include_symbols):
    global base_kernel_chars, char_to_long_char, char_to_num, num_to_char, gp_params_dim, param_idxs, vocabulary_size
    if include_scale:
        raise NotImplementedError()
        base_kernel_chars = {"W", "R", "E", "L", "1", "2", "3", "4", "5", "a", "b", "c", "d", "e"}
        char_to_long_char = {"W": "WN", "R": "SE", "E": "Per", "L": "Lin", "1":"Per1", "2":"Per2", "3":"Per3", "4":"Per4", "5":"Per5", "a":"Cos1", "b":"Cos2", "c":"Cos3", "d":"Cos4", "e":"Cos5"}
        char_to_num = {"+":0, "*":1, "W":2, "R":3, "E":4, "L":5, "1":6, "2":7, "3":8, "4":9, "5":10, "a":11, "b":12, "c":13, "d":14, "e":15}
        num_to_char = dict([(v, k) for k, v in char_to_num.items()])
        gp_params_dim = 8
    else:
        base_kernel_chars = {"W", "R", "C", "p", "1", "2", "3", "4", "5", "x", "a", "b", "c", "d", "e", "l", "!", "@", "#", "$", "%"}
        base_kernel_chars = {x for x in base_kernel_chars if x in include_symbols}
        char_to_long_char = {"W": "WN", "R": "SE", "C": "const", "p":"Per", "1":"Per1", "2":"Per2", "3":"Per3", "4":"Per4", "5":"Per5", "x":"Cos", "a":"Cos1", "b":"Cos2", "c":"Cos3", "d":"Cos4", "e":"Cos5", "l":"Lin", "!":"Lin1", "@":"Lin2", "#":"Lin3", "$":"Lin4", "%":"Lin5"}
        char_to_long_char = {k:v for k,v in char_to_long_char.items() if k in include_symbols}
        char_to_num = {k:i for i,k in enumerate(["+", "*", *sorted(base_kernel_chars)])}
        num_to_char = dict([(v, k) for k, v in char_to_num.items()])

        num_params = {"W":0, "R":1, "C":1, **{k:2 for k in "p12345"}, **{k:1 for k in "xabcdef"}, **{k:1 for k in "l!@#$%"}}
        param_idxs = {}
        gp_params_dim = 0
        for k, v in num_params.items():
            param_idxs[k] = [gp_params_dim+i for i in range(v)]
            gp_params_dim += v

    vocabulary_size = len(char_to_num)

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
            if not isinstance(param, list):
                param = [param]

            params_idx += 1
            if char in ["W", "R", "C", "p", "1", "2", "3", "4", "5", "x", "a", "b", "c", "d", "e", "l", "!", "@", "#", "$", "%"]:
                long_expression += "{}({})".format(
                    char_to_long_char[char],
                    ", ".join(f"{p:.2f}" for p in param) 
                )
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

    def __init__(self, expression, raw_params, mean_prior_sd=0.1):
        super().__init__()
        self.expression = expression
        self.index = 0

        self.raw_params = raw_params
        self.raw_params_index = 0
        self.params = []

        self.value = self.getValue()
        self.mean_prior_sd = mean_prior_sd

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
            elif char in ["E", "1", "2", "3", "4", "5"]:
                raise NotImplementedError()
                period_limits = (1/128, 1) if char=="E" else ((int(char)-1)/4 * 127/128+1/128, int(char)/4 * 127/128+1/128)
                scale_sq = F.softplus(raw_param[3])*0.1
                period = period_limits[0] + torch.sigmoid(raw_param[4])*(period_limits[1]-period_limits[0])
                lengthscale_sq = F.softplus(raw_param[5])
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
            raise NotImplementedError
        else:
            p = [raw_param[i] for i in param_idxs.get(char, [])]
            if char == "W":
                self.params.append([])
                return {"op": "WhiteNoise"}
            elif char == "R":
                lengthscale_sq = F.softplus(p[0])
                self.params.append(lengthscale_sq.item())
                return {"op": "RBF", "lengthscale_sq": lengthscale_sq}
            elif char in ["p", "1", "2", "3", "4", "5"]:
                if char == "p":
                    period_limits = (1/32, 1)
                elif char=="1":
                    period_limits = (1/32, 1/16)
                elif char=="2":
                    period_limits = (1/16, 1/8)
                elif char=="3":
                    period_limits = (1/8, 1/4)
                elif char=="4":
                    period_limits = (1/4, 1/2)
                elif char=="5":
                    period_limits = (1/2, 1)

                # period_limits = (1/128, 1) if char=="E" else ((int(char)-1)/4 * 127/128+1/128, int(char)/4 * 127/128+1/128)
                # period = torch.sigmoid(raw_param[1] - 1)
                period = period_limits[0] + torch.sigmoid(p[0])*(period_limits[1]-period_limits[0])
                lengthscale_sq = F.softplus(p[1]) * period
                self.params.append([period.item(), lengthscale_sq.item()])
                return {"op": "ExpSinSq", "period": period, "lengthscale_sq": lengthscale_sq, }
            elif char in ["x", "a", "b", "c", "d", "e"]:
                if char=="x":
                    period_limits = (1/32, 1)
                if char=="a":
                    period_limits = (1/32, 1/16)
                elif char=="b":
                    period_limits = (1/16, 1/8)
                elif char=="c":
                    period_limits = (1/8, 1/4)
                elif char=="d":
                    period_limits = (1/4, 1/2)
                elif char=="e":
                    period_limits = (1/2, 1)
                # period_limits = (1/128, 1) if char=="E" else ((int(char)-1)/4 * 127/128+1/128, int(char)/4 * 127/128+1/128)
                # period = torch.sigmoid(raw_param[1] - 1)
                period = period_limits[0] + torch.sigmoid(p[0])*(period_limits[1]-period_limits[0])
                self.params.append([period.item()])
                return {"op": "Cosine", "period": period}
            # elif char == "L":
            #     offset = raw_param[16] + 0.5
            #     self.params.append(offset.item())
            #     return {"op": "Linear",  "offset": offset}  
            elif char in ["l", "!", "@", "#", "$", "%"]:
                if char=="l":
                    offset_limits = (-1.5, 1.5)
                if char=="!":
                    offset_limits = (-1.5, -0.5)
                elif char=="@":
                    offset_limits = (-0.5, 0.25)
                elif char=="#":
                    offset_limits = (0.25, 0.75)
                elif char=="$":
                    offset_limits = (0.75, 1.5)
                elif char=="%":
                    offset_limits = (1.5, 2.5)
                offset = offset_limits[0] + torch.sigmoid(p[0])*(offset_limits[1]-offset_limits[0])
                self.params.append(offset.item())
                return {"op": "Linear",  "offset": offset}  
            elif char == "C":
                value = F.softplus(p[0])
                self.params.append(value.item())
                return {"op": "Constant", "value": value}

        raise ParsingError("Cannot parse char: " + str(char))
        

    def forward(self, x_1, x_2, kernel=None, top_level=True):
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
                t = t + self.forward(x_1, x_2, v, top_level=False)
            result = t
        elif kernel["op"] == "*":
            t = self.forward(x_1, x_2, kernel["values"][0], top_level=False)
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
        elif kernel["op"] == "Cosine":
            t = kernel["period"]
            s2 = kernel.get("scale_sq", 1)
            result = (2 * math.pi * (x_1 - x_2).abs() / t).cos() * s2
            # print(kernel['op'], "t", t, "l2", l2, "s2", s2)
        elif kernel["op"] == "Constant":
            c = kernel["value"]
            result = torch.full((x_1-x_2).shape, c.item(), device=x_1.device)
        else:
            raise ParsingError(f"Cannot parse kernel op {kernel['op']}")

        # Test nan
        if torch.isnan(result).any():
            raise RuntimeError("Covariance nan")

        if top_level:
            result = result + self.mean_prior_sd**2

        return result


# Init, saving, etc
def init(run_args, device, fast=False):
    memory = None
    if run_args.model_type == "timeseries":
        init_symbols(run_args.include_symbols)

        # Generative model
        generative_model = timeseries.GenerativeModel(
            max_num_chars=run_args.max_num_chars, lstm_hidden_dim=run_args.generative_model_lstm_hidden_dim,
	    learn_eps=getattr(run_args, 'learn_eps', False)
        ).to(device)

        # Guide
        guide = timeseries.Guide(
            max_num_chars=run_args.max_num_chars, lstm_hidden_dim=run_args.guide_lstm_hidden_dim
        ).to(device)

        if not fast:
            # Pretrain the prior
            cmws.examples.timeseries.expression_prior_pretraining.pretrain_expression_prior(
                generative_model, guide, batch_size=10, num_iterations=4000, include_symbols=run_args.include_symbols or None
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
    if hasattr(run_args, "lr_guide_continuous"): 
        guide_continuous_params = [*guide.gp_params_lstm.parameters(), *guide.gp_params_extractor.parameters()]
        guide_discrete_params = [
            *guide.obs_embedder.parameters(),
            *guide.expression_embedder.parameters(),
            *guide.expression_lstm.parameters(),
            *guide.expression_extractor.parameters(),
        ]
    
        prior_continuous_params = [*generative_model.gp_params_lstm.parameters(), *generative_model.gp_params_extractor.parameters()]
        prior_discrete_params = [*generative_model.expression_lstm.parameters(), *generative_model.expression_extractor.parameters()]
        likelihood_params = [generative_model.log_eps] if generative_model.learn_eps else []
    
        assert len(set(guide.parameters()) - set([*guide_continuous_params, *guide_discrete_params])) == 0
        assert len(set(generative_model.parameters()) - set([*prior_continuous_params, *prior_discrete_params, *likelihood_params])) == 0

        optimizer = torch.optim.Adam([
            {'params': guide_continuous_params, 'lr':run_args.lr_guide_continuous},
            {'params': guide_discrete_params, 'lr':run_args.lr_guide_discrete},
            {'params': prior_continuous_params, 'lr':run_args.lr_prior_continuous},
            {'params': prior_discrete_params, 'lr':run_args.lr_prior_discrete},
            {'params': likelihood_params, 'lr':run_args.lr_likelihood},
        ], lr=run_args.lr)

    elif hasattr(run_args, "lr_continuous_latents"):
        #LEGACY CODE
        continuous_latent_parameters = list(guide.gp_params_extractor.parameters())
        other_parameters = [
            *generative_model.parameters(),
            *(set(guide.parameters()) - set(continuous_latent_parameters))
        ]

        optimizer = torch.optim.Adam([
            {'params': continuous_latent_parameters, 'lr':run_args.lr_continuous_latents},
            {'params': other_parameters},
        ], lr=run_args.lr)
    else:
        raise NotImplementedError()

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
