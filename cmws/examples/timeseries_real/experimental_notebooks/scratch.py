from pathlib import Path

from cmws import train, util
from cmws.examples.timeseries import util as timeseries_util


from cmws.examples.timeseries.run import main, get_args_parser

parser = get_args_parser()
args = parser.parse_args([
    "--continue-training",
    "--experiment-name=scratch4",
    "--algorithm=cmws_5",
    "--num-particles=25",
    "--full-training-data",
    "--generative-model-lstm-hidden-dim=10",
    "--guide-lstm-hidden-dim=10",
    "--memory-size=5",
    "--num-proposals-mws=15",
#     "--num-sleep-pretraining-iterations=1000",
    "--max-num-chars=7",
    "--lr=0.001",
    "--lr-continuous-latents=0.02",
    "--lr-sleep-pretraining=0.01"#,
#     "--fast"
])
main(args)
