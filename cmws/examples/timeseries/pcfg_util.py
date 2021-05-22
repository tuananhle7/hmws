import itertools
import json

import torch


def read_pcfg(pcfg_path, device, include_symbols=None):
    with open(pcfg_path) as json_data:
        data = json.load(json_data)

    if include_symbols is not None:
        include_symbols = include_symbols + "+*KT"
        if any([x not in include_symbols for x in data['terminals']]):
            print("Ignoring terminals: ", [x for x in data['terminals'] if x not in include_symbols])
            data['terminals'] = [x for x in data['terminals'] if x in include_symbols]
        if any([x not in include_symbols for x in data['non_terminals']]):
            print("Ignoring non-terminals: ", [x for x in data['non_terminals'] if x not in include_symbols])
            data['non_terminals'] = [x for x in data['non_terminals'] if x in include_symbols]

        for k in data['productions']:
            data['production_probs'][k] = [prob
                                           for prod, prob in zip(data['productions'][k], data['production_probs'][k])
                                           if all(x in include_symbols for x in prod)]
            data['productions'][k] = [prod for prod in data['productions'][k]
                                      if all(x in include_symbols for x in prod)]

    grammar = {
        "terminals": set(data["terminals"]),
        "non_terminals": set(data["non_terminals"]),
        "productions": data["productions"],
        "start_symbol": data["start_symbol"],
        "name": data["name"],
    }
    production_probs = production_probs_to_tensor(data["production_probs"], device)

    return grammar, production_probs


def production_probs_to_tensor(production_probs, device):
    """Convert production_probs in list to tensor.

    Args:
        production_probs: dict whose keys are non-terminals and values are
            probabilities of productions represented as list of shape
            [num_productions]
    Returns: same as production_probs but values are tensors instead of
    lists.
    """
    return {
        k: torch.tensor(v, dtype=torch.float, device=device)/sum(v) for k, v in production_probs.items()
    }


def sample_tree(pcfg, symbol=None, depth=0, max_depth=20):
    """Sample tree from prior.
    Args:
        pcfg
            grammar
            production_probs
        symbol (str)
        depth (int)
        max_depth (int)
    Returns: list of lists or string
    """
    grammar, production_probs = pcfg

    if symbol is None:
        symbol = grammar["start_symbol"]

    if symbol in grammar["terminals"]:
        return symbol
    elif depth > max_depth:
        raise RuntimeError("Max depth reached")
    else:
        production_index = torch.distributions.Categorical(probs=production_probs[symbol]).sample()
        production = grammar["productions"][symbol][production_index]
        return [symbol] + [
            sample_tree(pcfg, s, depth=depth + 1, max_depth=max_depth) for s in production
        ]


def get_leaves(tree):
    """Return leaves of a tree.
    Args: list of lists or string
    Returns: list of strings
    """
    if isinstance(tree, list):
        return list(itertools.chain.from_iterable([get_leaves(subtree) for subtree in tree[1:]]))
    else:
        return [tree]


def sample_expression(pcfg):
    return "".join(get_leaves(sample_tree(pcfg)))


if __name__ == "__main__":
    device = "cpu"
    pcfg = read_pcfg("kernel_pcfg.json", device)
    print(sample_expression(pcfg))
