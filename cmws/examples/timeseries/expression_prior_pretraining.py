import itertools
import pathlib

import cmws
import cmws.examples.timeseries.lstm_util as lstm_util
import cmws.examples.timeseries.pcfg_util as pcfg_util
import cmws.examples.timeseries.util as timeseries_util
import torch
from tqdm import tqdm


def generate_data(batch_size, max_num_chars, pcfg, device, verbose=False):
    x = torch.zeros((batch_size, max_num_chars), device=device).long()
    eos = torch.zeros((batch_size, max_num_chars), device=device).long()

    for batch_id in range(batch_size):
        done = False
        while not done:
            expr = pcfg_util.sample_expression(pcfg)
            if verbose:
                cmws.util.logging.info(expr)
            raw_expression = timeseries_util.get_raw_expression(expr, device)
            expression_len = len(raw_expression)
            if expression_len > max_num_chars:
                continue
            x[batch_id, :expression_len] = raw_expression
            eos[batch_id, expression_len - 1] = 1
            done = True
    return x, eos


@torch.enable_grad()
def pretrain_expression_prior(generative_model, guide, batch_size, num_iterations, include_symbols):
    cmws.util.logging.info("Pretraining the expression prior")
    optimizer = torch.optim.Adam(
        itertools.chain(
            generative_model.expression_lstm.parameters(),
            generative_model.expression_extractor.parameters(),
            guide.expression_lstm.parameters(),
            guide.expression_extractor.parameters(),            
        )
    )
    train_data_iterator = cmws.util.cycle(
        cmws.examples.timeseries.data.get_timeseries_data_loader(
            generative_model.device, batch_size, test=False, full_data=True
        )
    )

    path = pathlib.Path(__file__).parent.absolute().joinpath("kernel_pcfg_coarse_ordered.json")
    pcfg = pcfg_util.read_pcfg(path, generative_model.device, include_symbols=include_symbols)
    for i in tqdm(range(num_iterations)):
        x, eos = generate_data(batch_size, generative_model.max_num_chars, pcfg, generative_model.device, verbose=i==0)
        optimizer.zero_grad()
        
        loss = -generative_model.expression_dist.log_prob(x, eos).mean()

        obs, obs_id = next(train_data_iterator)
        loss = loss - guide.log_prob_discrete(obs, (x, eos)).mean()

        loss.backward()
        optimizer.step()


def sample_expression(generative_model, num_samples):
    x, eos = generative_model.expression_dist.sample([num_samples])
    num_timesteps = lstm_util.get_num_timesteps(eos)
    expression = []
    for i in range(num_samples):
        expression.append(timeseries_util.get_expression(x[i, : num_timesteps[i]]))
    return expression
