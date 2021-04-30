import math

import torch
import torch.nn as nn
import torch.nn.functional as F

base_kernel_chars = {"W", "R", "E", "C"}
char_to_num = {"W": 0, "R": 1, "E": 2, "C": 3, "*": 4, "+": 5, "(": 6, ")": 7}
num_to_char = dict([(v, k) for k, v in char_to_num.items()])
vocabulary_size = len(char_to_num)
gp_params_dim = 5


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


class Kernel(nn.Module):
    """Kernel -> Kernel + Kernel | Kernel * Kernel | W | R | E | C
    Adapted from
    https://github.com/insperatum/wsvae/blob/7dee0708587e6a33b7328206ce5edd8262d568b6/gp.py#L12

    Args
        expression (str) consists of characters *, +, (, ) or
            W = WhiteNoie
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

    def getValue(self):
        value = self.parseExpression()
        if self.hasNext():
            raise Exception(
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
                raise Exception("No closing parenthesis found at character " + str(self.index))
            self.index += 1
            return value
        else:
            return self.parseValue()

    def parseValue(self):
        char = self.peek()
        self.index += 1
        param = self.raw_params[self.raw_params_index]
        self.raw_params_index += 1
        if char == "W":
            return {"op": "WhiteNoise", "scale_sq": F.softplus(param[0])}
        elif char == "R":
            return {"op": "RBF", "lengthscale_sq": F.softplus(param[1])}
        elif char == "E":
            return {
                "op": "ExpSinSq",
                "period": F.softplus(param[2]),
                "lengthscale_sq": F.softplus(param[3]),
            }
        if char == "C":
            return {"op": "Constant", "const": F.softplus(param[4])}
        else:
            raise NotImplementedError("Cannot parse char: " + str(char))

    def forward(self, x_1, x_2, kernel=None):
        """
        Args
            x_1 [batch_size, num_timesteps_1, 1]
            x_2 [batch_size, 1, num_timesteps_2]

        Returns [batch_size, num_timesteps_1, num_timesteps_2]
        """
        # batch_size = x_1.size(0)
        n = x_1.size(1)
        if kernel is None:
            kernel = self.value
        if kernel["op"] == "+":
            t = self.forward(x_1, x_2, kernel["values"][0])
            for v in kernel["values"][1:]:
                t += self.forward(x_1, x_2, v)
            return t
        elif kernel["op"] == "*":
            t = self.forward(x_1, x_2, kernel["values"][0])
            for v in kernel["values"][1:]:
                t *= self.forward(x_1, x_2, v)
            return t
        elif kernel["op"] == "Constant":
            return kernel["const"].repeat(1, n)
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
            raise NotImplementedError()
