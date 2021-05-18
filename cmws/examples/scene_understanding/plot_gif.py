from cmws import util

root_dir = "save/2021_05_13_cmws_vs_rws/SU_cmws_2_0"
util.make_gif(
    [f"{root_dir}/reconstructions/{i * 100 + 1}.png" for i in range(62)],
    f"{root_dir}/reconstruction.gif",
    10,
)
