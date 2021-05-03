import cmws.examples.timeseries.pcfg_util as pcfg_util
import cmws
import cmws.examples.timeseries.util as timeseries_util
import torch
import itertools
from tqdm import tqdm
import cmws.examples.timeseries.lstm_util as lstm_util


def generate_data(batch_size, max_num_chars, device):
    pcfg = pcfg_util.read_pcfg("kernel_pcfg.json", device)

    x = torch.zeros((batch_size, max_num_chars), device=device).long()
    eos = torch.zeros((batch_size, max_num_chars), device=device).long()

    for batch_id in range(batch_size):
        done = False
        while not done:
            raw_expression = timeseries_util.get_raw_expression(
                pcfg_util.sample_expression(pcfg), device
            )
            expression_len = len(raw_expression)
            if expression_len > max_num_chars:
                continue
            x[batch_id, :expression_len] = raw_expression
            eos[batch_id, expression_len - 1] = 1
            done = True
    return x, eos


def pretrain_expression_prior(generative_model, batch_size, num_iterations):
    cmws.util.logging.info("Pretraining the expression prior")
    optimizer = torch.optim.Adam(
        itertools.chain(
            generative_model.expression_lstm.parameters(),
            generative_model.expression_extractor.parameters(),
        )
    )
    for i in tqdm(range(num_iterations)):
        x, eos = generate_data(batch_size, generative_model.max_num_chars, generative_model.device)

        optimizer.zero_grad()
        loss = -generative_model.expression_dist.log_prob(x, eos).mean()
        loss.backward()
        optimizer.step()


def sample_expression(generative_model, num_samples):
    x, eos = generative_model.expression_dist.sample([num_samples])
    num_timesteps = lstm_util.get_num_timesteps(eos)
    expression = []
    for i in range(num_samples):
        expression.append(timeseries_util.get_expression(x[i, : num_timesteps[i]]))
    return expression
