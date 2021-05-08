import argparse
import getpass
import shutil
import socket
import subprocess
from pathlib import Path

import cmws


def cancel_all_my_non_bash_jobs():
    cmws.util.logging.info("Cancelling all non-bash jobs.")
    jobs_status = (
        subprocess.check_output(f"squeue -u {getpass.getuser()}", shell=True)
        .decode()
        .split("\n")[1:-1]
    )
    non_bash_job_ids = []
    for job_status in jobs_status:
        if not ("bash" in job_status.split() or "zsh" in job_status.split()):
            non_bash_job_ids.append(job_status.split()[0])
    if len(non_bash_job_ids) > 0:
        cmd = "scancel {}".format(" ".join(non_bash_job_ids))
        cmws.util.logging.info(cmd)
        cmws.util.logging.info(subprocess.check_output(cmd, shell=True).decode())
    else:
        cmws.util.logging.info("No non-bash jobs to cancel.")


def args_to_str(args):
    result = ""
    for k, v in vars(args).items():
        k_str = k.replace("_", "-")
        if v is None:
            pass
        elif isinstance(v, bool):
            if v:
                result += " --{}".format(k_str)
        else:
            if isinstance(v, list):
                v_str = " ".join(map(str, v))
            else:
                v_str = v
            result += " --{} {}".format(k_str, v_str)
    return result


def submit_slurm_job(run_args, logs_dir, job_name, no_repeat=False):
    # SBATCH AND PYTHON CMD
    args_str = args_to_str(run_args)
    if no_repeat:
        sbatch_cmd = "sbatch"
        time_option = "12:0:0"
        python_cmd = f'--wrap="MKL_THREADING_LAYER=INTEL=1 python -u run.py {args_str}""'
    else:
        sbatch_cmd = "om-repeat sbatch"
        time_option = "12:0:0"
        python_cmd = f"MKL_THREADING_LAYER=INTEL=1 python -u run.py {args_str}"

    # SBATCH OPTIONS
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    sbatch_options = (
        f"--time={time_option} "
        + "--ntasks=1 "
        + f"--gres=gpu:1 "
        # + "--constraint=high-capacity "
        # + f"-p tenenbaum "
        + "--constraint=14GB "
        + "-x node093 "
        + f'-J "{job_name}" '
        + f'-o "{logs_dir}/%j.out" '
        + f'-e "{logs_dir}/%j.err" '
    )
    cmd = " ".join([sbatch_cmd, sbatch_options, python_cmd])
    cmws.util.logging.info(cmd)
    subprocess.call(cmd, shell=True)


def submit_slurm_jobs(
    get_run_argss, get_config_name, get_job_name, no_repeat=False, cancel=False, rm=False
):
    if cancel:
        cancel_all_my_non_bash_jobs()

    if rm:
        dir_ = f"save/{next(iter(get_run_argss())).experiment_name}"
        if Path(dir_).exists():
            shutil.rmtree(dir_, ignore_errors=True)

    cmws.util.logging.info(
        f"Launching {len(list(get_run_argss()))} runs from {socket.gethostname()}"
    )
    for run_args in get_run_argss():
        config_name = get_config_name(run_args)
        job_name = get_job_name(run_args)
        submit_slurm_job(
            run_args,
            cmws.util.get_logs_dir(run_args.experiment_name, config_name),
            job_name,
            no_repeat=no_repeat,
        )


def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--local", action="store_true", help="")
    parser.add_argument("--rm", action="store_true", help="")
    parser.add_argument(
        "--no-repeat",
        action="store_true",
        help="run the jobs using standard sbatch."
        "if False, queues 2h jobs with dependencies "
        "until the script finishes",
    )
    parser.add_argument(
        "--cancel", action="store_true", help="cancels all non-bash jobs if run on the cluster"
    )
    return parser
