from __future__ import print_function
import sys

import os as _os
import re as _re
import subprocess as _subprocess
import sys as _sys
from collections import namedtuple as _namedtuple
from copy import deepcopy as _deepcopy
import constants
import timing as _timing


PartitionParams = _namedtuple("PartitionParams",
                              ["name", "num_cpus", "num_gpus", "timeout"])

if sys.version_info[:2] < (3, 3):
    old_print = print
    def print(*args, **kwargs):
        flush = kwargs.pop('flush', False)
        old_print(*args, **kwargs)
        if flush:
            file = kwargs.get('file', sys.stdout)
            # Why might file=None? IDK, but it works for print(i, file=None)
            file.flush() if file is not None else sys.stdout.flush()
    from pipes import quote
else:
    from shlex import quote


def get_device_list(exp):
    """ Returns set of GPU devices local to the rank"""

    visible_device_list = _os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    partition = exp.scheduling_config.partition
    partition_params = get_partition_params(partition)
    max_gpus_per_node = partition_params.num_gpus
    num_ranks = exp.scheduling_config.num_ranks
    num_gpus_per_rank = exp.scheduling_config.num_gpus_per_rank
    tot_num_gpus = num_gpus_per_rank * num_ranks
    num_nodes = (tot_num_gpus + max_gpus_per_node - 1) // max_gpus_per_node
    num_ranks_per_node = (num_ranks + num_nodes - 1) // num_nodes
    local_rank = constants.MPI_RANK % num_ranks_per_node
    start_id = local_rank * num_gpus_per_rank
    this_rank_device_list = ','.join(str(i) for i in
                                     range(start_id,
                                     start_id + num_gpus_per_rank))
    return this_rank_device_list


def get_partition_params(partition):
    """Return slurm partition architecture parameters.

    Args:
        partition (str): name of partition

    Returns:
        params (PartitionParams): namedtuple with attributes
            - name: partition name
            - num_cpus: total number of cpus per node
            - num_gpus: total number of gpus per node
            - timeout: max time to run, in secs.
    Raises:
        PartitionError: If unable to find params for given partition.
    """

    def parse_timelimit(timeout):
        timeout = timeout.split("-")
        total_secs = 0
        if len(timeout) == 2:   # includes days
            days = timeout.pop(0)
            total_secs += 60 * 60 * float(days)
        time_chunks = timeout[0].split(":")
        if len(time_chunks) > 2:
            hrs, mins, secs = map(float, time_chunks)
        else:
            hrs = 0
            mins, secs = map(float, time_chunks)
        total_secs += hrs * 60 * 60 + mins * 60 + secs
        return total_secs

    output = _subprocess.check_output(['sinfo', '-o', '"%P %c %G %l"'])
    output = output.decode("utf-8")
    output = output.strip('"\n"')
    partitions = output.split('"\n"')

    # e.g., "TitanXx8 40 gpu:8"
    # e.g., "GTX980x4short* 32 gpu:4"
    pattern = _re.compile(r'(\S+) (\d+) gpu:(\d+) ((\d\-)?(\d+\:)?\d+\:\d+)')

    for line in partitions:

        match = pattern.match(line)
        if not match:
            continue
        partition_name, num_cpus, num_gpus, timeout = match.group(1, 2, 3, 4)

        total_secs = parse_timelimit(timeout)

        partition_name = partition_name.rstrip('*')  # strip star
        if partition_name == partition:
            if "Pascal" in partition or "1080Ti" in partition:
                # TODO: fix this at the system level by getting Slurm to
                # report the correct number of CPUs.
                # These two paritions only have 20 CPUS though reporting 40.
                # In addition, if user makes a reservation of any of these
                # two partitions, the partition name will become "Pascal*" or
                # "1080Ti*"
                num_cpus_this_partition = 20
            else:
                num_cpus_this_partition = int(num_cpus)
            return PartitionParams(name=partition_name,
                                   num_cpus=num_cpus_this_partition,
                                   num_gpus=int(num_gpus),
                                   timeout=total_secs)

    raise Exception("Unable to find partition params for {}."
                    .format(partition))


def slurm_restart(checkpoint_helper, should_restart, *args):
    """Checkpoint and enqueue a SLURM job through the first rank.
    Note: All the ranks handle USR1, but only use this to exit gracefully
    Args:
        checkpoint_helper (nlm.utils.CheckpointHelper)
        should_restart (bool): Jobs are only restarted if this is True
            Only checkpoint otherwise
        *args: Used to accommodate args passed in for handling signals
    """
    print("Rank {} received restart request @ {}" \
          .format(constants.MPI_RANK, _timing.curr_time()), flush=True)
    try:
        checkpoint_helper.checkpoint()
    except Exception as e:
        print("ERROR: Checkpointing at the end of the job failed", flush=True)
        print("{}".format(str(e)), flush=True)

    if constants.MPI_RANK_0 and should_restart:
        try:
            restart_job()
        except Exception as e:
            print("ERROR: Restarting failed", flush=True)
            print("{}".format(str(e)), flush=True)
    _sys.exit(0)


def slurm_restart_slim(dir_exist, should_restart, *args):
    """Checkpoint and enqueue a SLURM job through the first rank.
    Note: All the ranks handle USR1, but only use this to exit gracefully
    Args:
        checkpoint_helper (nlm.utils.CheckpointHelper)
        should_restart (bool): Jobs are only restarted if this is True
            Only checkpoint otherwise
        *args: Used to accommodate args passed in for handling signals
    """
    print("Rank {} received restart request @ {}" \
          .format(constants.MPI_RANK, _timing.curr_time()), flush=True)

    if not dir_exist:
        print("ERROR: No check point at the end of the job failed", flush=True)

    if constants.MPI_RANK_0 and should_restart:
        try:
            restart_job()
        except Exception as e:
            print("ERROR: Restarting failed", flush=True)
            print("{}".format(str(e)), flush=True)
    _sys.exit(0)


def restart_job():
    """Restart this job and enqueue it through SLURM
    """

    # Don't restart if we're not MPI rank zero or equivalent
    if "PMI_RANK" in _os.environ and _os.environ["PMI_RANK"] != "0":
        return

    # SIGKILL can be sent for many reasons. Check that we're over the time
    # limit; if we are, assume it's a timeout. Consider it a timeout if the
    # runtime is within some buffer seconds of the limit.
    # job_info = get_slurm_job_info()
    restart_cmd = " ".join(create_sbatch_restart_command())
    # run_time = job_info["RunTime"]
    # time_limit = job_info["TimeLimit"]

    print("Timed out. Restarting SLURM job...", flush=True)
    print("Restarting: {}".format(restart_cmd), flush=True)

    output = str(_subprocess.check_output(restart_cmd, shell=True))
    job_id = _re.search(r"[Jj]ob [0-9]+", output).group(0)
    print("Submitted new job: {}".format(job_id), flush=True)

    # Exit immediately
    _sys.exit(0)


def restart_job_command(restart_command):
    # Don't restart if we're not MPI rank zero or equivalent
    if "PMI_RANK" in _os.environ and _os.environ["PMI_RANK"] != "0":
        return

    print("Timed out. Restarting SLURM job...", flush=True)
    print("Restarting: {}".format(restart_command), flush=True)

    output = str(_subprocess.check_output(restart_command, shell=True))
    job_id = _re.search(r"[Jj]ob [0-9]+", output).group(0)
    print("Submitted new job: {}".format(job_id), flush=True)

    # Exit immediately
    _sys.exit(0)



def get_slurm_job_info():
    """Get information about the current job using `scontrol show job`.

    Returns a dict mapping parameter names (e.g. "UserId", "RunTime", etc) to
    their values, both as strings.
    """
    job_id = int(_os.environ["SLURM_JOB_ID"])

    command = ["scontrol", "show", "job", str(job_id)]
    output = _subprocess.check_output(command).decode("utf-8")

    # Use a regex to extract the parameter names and values
    pattern = "([A-Za-z/]*)=([^ \t\n]*)"
    return dict(_re.findall(pattern, output))


def create_sbatch_restart_command():
    """Using the environment and SLURM command, create a command that, when,
    run, will enqueue a repeat of the current job using `sbatch`.

    Return the command as a list of strings, suitable for passing to
    `subprocess.check_call` or similar functions.
    """
    # Get all the necessary information by querying SLURM with this job id
    info = get_slurm_job_info()
    partition = info.get("Partition")
    # partition = "TitanXx8,TitanXx8_slong,1080Ti,TitanXx8_slong,1080Ti_slong,M40x8,M40x8_slong"

    num_cpus = int(info["NumCPUs"])
    cpus_per_task = int(info["CPUs/Task"])
    gres = info.get("Gres")
    stderr = info.get("StdErr")
    stdout = info.get("StdOut")
    num_nodes = int(info["NumNodes"])
    reservation = info.get("Reservation")

    command = ["sbatch"]

    tasks = num_cpus // cpus_per_task
    tasks_per_node = (tasks + num_nodes - 1) // num_nodes
    command.extend(["-N", str(num_nodes)])
    command.extend(["--ntasks", str(tasks)])
    command.extend(["--ntasks-per-node", str(tasks_per_node)])
    command.extend(["--cpus-per-task", str(cpus_per_task)])
    command.extend(["--open-mode", "append"])
    command.extend(["--signal", "USR1@600"])

    argv = _deepcopy(_sys.argv)

    if "EXPERIMENT_ID" in _os.environ:
        curr_restart_count = _os.environ.get("RESTART_COUNT", '-1')
        new_restart_count = str(int(curr_restart_count) + 1)
        new_name = "{}_{}".format(_os.environ["EXPERIMENT_ID"],
                                  new_restart_count)
        command.extend(["--job-name ", new_name])

    if partition:
        command.extend(["--partition", partition])
    if reservation and reservation != "(null)":
        command.extend(["--reservation", reservation])
    if gres and gres != "(null)":
        command.extend(["--gres", gres])
    if stderr:
        command.extend(["--error", stderr])
    if stdout:
        command.extend(["--output", stdout])

    which = "/usr/bin/which"
    python = _subprocess.check_output(
        [which, "python"]).decode("utf-8").strip()
    wrap_cmd = ["srun", python] + argv
    command.extend(["--wrap \"",
                    " ".join(quote(arg) for arg in wrap_cmd),
                    " \""])

    return command


def create_sbatch_enqueue_command(mode, exp_dir, partition, exp_config=None,
                                  exp_name=None, data=None, num_ranks=1,
                                  num_gpus_per_rank=1,
                                  reservation=None, sbatch_args=None,
                                  tokens=None, batch_size=256, vocab=None,
                                  max_len=None, load_mode=None):
    """Create a job SLURM enqueue command
    Args:
        mode (str): train/eval
        exp_dir (str): Path to model directory
        partition (str): SLURM partition to use
        exp_config (str): Path to experiment configuration
        exp_name (str): Name of the experiment
        data (str): Path to the dataset
        num_ranks (int): Number of data parallel workers
        num_gpus_per_rank (int): Number of GPUs per data parallel worker
        reservation (str): SLURM reservation
        tokens (str): dataset.TOKEN_TYPES
        batch_size (int)
        vocab (str): Path to vocabulary
        load_mode (str): Load the latest/best checkpoint
    """
    partition_params = get_partition_params(partition)
    max_gpus_per_node = partition_params.num_gpus
    tot_num_gpus = num_gpus_per_rank * num_ranks
    gpus_per_node = min(max_gpus_per_node, tot_num_gpus)
    num_nodes = (tot_num_gpus + max_gpus_per_node - 1) // max_gpus_per_node
    num_tasks_per_node = (num_ranks + num_nodes - 1) // num_nodes
    tot_num_cpus = partition_params.num_cpus
    cpus_per_task = tot_num_cpus // max_gpus_per_node * num_gpus_per_rank

    if batch_size % num_ranks != 0:
        print("ERROR: batchSize: {} is not a multiple of "
              "data parallel workers: {}\n".format(batch_size, num_ranks))
        _sys.exit(0)

    if tot_num_gpus > max_gpus_per_node and\
            max_gpus_per_node % num_gpus_per_rank != 0:
        print("ERROR: max_gpus_per_node: {} is not a multiple of "
              "num_gpus_per_rank:{}\n".format(max_gpus_per_node,
                                              num_gpus_per_rank), flush=True)
        _sys.exit(0)

    if reservation is not None:
        reservation_str = '--reservation={}'.format(reservation)
    else:
        reservation_str = ''

    if sbatch_args is not None:
        sbatch_args_str = '{}'.format(sbatch_args)
    else:
        sbatch_args_str = ''

    if mode == 'train':
        script_args = '--exp_config {exp_config} --exp_name {exp_name} '
        script_args = script_args.format(exp_config=exp_config,
                                         exp_name=exp_name)
        script = 'train.py'
    else:
        script_args = ('--data {data} '
                       '--tokens {tokens} '
                       '--vocab {vocab} '
                       '--batch_size {batch_size} '
                       '--num_ranks={num_ranks} '
                       '--max_len={max_len} '
                       '--{load_mode}'
                       .format(data=data,
                               tokens=tokens, vocab=vocab,
                               batch_size=batch_size,
                               num_ranks=num_ranks,
                               max_len=max_len,
                               load_mode=load_mode))
        exp_name = 'eval'
        script = 'eval.py'

    command = (
        'sbatch -N {num_nodes} '
        '--ntasks-per-node {num_tasks_per_node} '
        '--job-name={exp_name} '
        '--ntasks {num_ranks} '
        '--cpus-per-task {cpus_per_task} '
        '--gres=gpu:{gpus_per_node} '
        '--signal=USR1@600 '
        '--wrap "srun stdbuf -i0 -o0 -e0 '
        'python {script} --exp_dir {exp_dir} ' +
        script_args +
        '" '
        '--partition={partition} '
        '{reservation} '
        '{sbatch_args}').format(
            num_nodes=num_nodes, exp_name=exp_name, num_ranks=num_ranks,
            cpus_per_task=cpus_per_task, gpus_per_node=gpus_per_node,
            script=script, exp_dir=exp_dir, partition=partition,
            reservation=reservation_str, sbatch_args=sbatch_args_str,
            num_tasks_per_node=num_tasks_per_node
    )
    return command


def get_worker_args():
    # Set up batch loading from the training and validation datasets.
    if constants.USING_MPI:
        worker_args = {"worker_index": constants.MPI_RANK,
                       "worker_count": constants.MPI_SIZE}
    else:
        worker_args = {"worker_index": 0,
                       "worker_count": 1}
    return worker_args


def change_slurm_jobname(job_id, new_jobname):
    """ Change slurm job's name.
    Args:
        job_id (int): Slurm job ID.
        new_jobname (str): New slurm job name.
    """
    command = 'scontrol update JobId={}  JobName=\"{}\"'.format(job_id,
                                                                new_jobname)
    output = _subprocess.call(command, stderr=_subprocess.STDOUT, shell=True)
    assert output == 0, "Couldn't change the job's name!"


def slurm_setup():
    # resolve %j.
    # directly manipulate sys.argv on purpose
    # as this is the first function to run, so no unexplained side effects.
    print("slurm setup")
    for i, s in enumerate(_sys.argv):
        print(_sys.argv[i], flush=True)
        if '%j' in s:
            _sys.argv[i] = _re.sub('%j', _os.environ['SLURM_JOB_ID'], s)

    restart_count = _os.environ.get("RESTART_COUNT", '-1')
    _os.environ["RESTART_COUNT"] = str(int(restart_count) + 1)

    if 'EXPERIMENT_ID' not in _os.environ:
        _os.environ["EXPERIMENT_ID"] = str(_os.environ['SLURM_JOB_ID'])

    new_name = "{}_{}".format(_os.environ["EXPERIMENT_ID"],
                              _os.environ["RESTART_COUNT"])
    change_slurm_jobname(_os.environ['SLURM_JOB_ID'], new_name)


def slurm_resolve(*paths):
    resolved_paths = []
    for path in paths:
        if path and '%j' in path:
            resolved = _re.sub('%j', _os.environ['SLURM_JOB_ID'], path)
        else:
            resolved = path
        resolved_paths.append(
            _os.path.expandvars(_os.path.expanduser(resolved)))
    return resolved_paths
