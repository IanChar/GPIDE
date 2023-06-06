"""
Launch several training jobs.

Author: Ian Char
Date: February 16, 2023
"""
import argparse
from dataclasses import dataclass
import datetime
from typing import Any, Dict
import os
import subprocess
import time

import numpy as np

from rlkit import RLKIT_PROJECT_PATH


# Parse Arguments.
parser = argparse.ArgumentParser()
parser.add_argument('--script', type=str,
                    default='scripts/train.py')
parser.add_argument('--name', type=str, default=None)
parser.add_argument('--rl', type=str,
                    default='msd_rnn_sac')
parser.add_argument('--env', type=str,
                    default='msd-fixed')
parser.add_argument('--num_seeds', type=int,
                    default=5)
parser.add_argument('--split_type', type=str,
                    default='te')
parser.add_argument('--job_log_dir', type=str,
                    default='jobs')
parser.add_argument('--num_gpus', type=int,
                    default=4)
parser.add_argument('--jobs_per_gpu', type=int,
                    default=2)
parser.add_argument('--curr_gpu_availability', type=str, default=None)
parser.add_argument('--seed_offset', type=int, default=None)
# Sweep some parameter should be structurd as "param1=x1,x2|param2=y1,y2,y3..."
parser.add_argument('--sweep', type=str, default=None)
args = parser.parse_args()

# Initialize global variables.
LOG_PATH = os.path.join(args.job_log_dir, f'{datetime.datetime.now()}.txt')
if args.curr_gpu_availability is not None:
    GPU_COUNTS = [int(g) for g in args.curr_gpu_availability.split(',')]
else:
    GPU_COUNTS = [0 for _ in range(args.num_gpus)]
MAX_RUNNING = sum([args.jobs_per_gpu - gc for gc in GPU_COUNTS])
RUNNING = []
if args.seed_offset is None:
    if args.split_type == 'tr':
        seed_offset = 100
    elif args.split_type == 'val':
        seed_offset = 200
    elif args.split_type == 'te':
        seed_offset = 300
    else:
        raise ValueError(f'Invalid split {args.split_type}')
else:
    seed_offset = args.seed_offset
ARG_DICT = {
    'seed': [i + seed_offset for i in range(args.num_seeds)],
    'env': args.env.split(','),
    'rl': args.rl.split(','),
    'split_type': [args.split_type],
}
if args.sweep is not None:
    sweep_list = args.sweep.split('|')
    for swp in sweep_list:
        k, vals = swp.split('=')
        ARG_DICT[k] = vals.split(',')


@dataclass
class Job:
    proc: subprocess.Popen
    gpu: int
    job_args: Dict[str, Any]


def prune_completed_job():
    for jidx, job in enumerate(RUNNING):
        if job.proc.poll() is not None:
            GPU_COUNTS[job.gpu] -= 1
            with open(LOG_PATH, 'a') as f:
                f.write(f'{datetime.datetime.now()}\t Finished \t {job.job_args}\n')
            RUNNING.pop(jidx)
            return True
    return False


def add_job(job_args):
    if len(RUNNING) >= MAX_RUNNING:
        while not prune_completed_job():
            time.sleep(30)
    with open(LOG_PATH, 'a') as f:
        f.write(f'{datetime.datetime.now()}\t Starting \t {job_args}\n')
    # Find open gpu device.
    gpu = 0
    while gpu < len(GPU_COUNTS) - 1 and GPU_COUNTS[gpu] >= args.jobs_per_gpu:
        gpu += 1
    GPU_COUNTS[gpu] += 1
    cmd = f'python {args.script} '
    prefix = args.name if args.name is not None else str(datetime.date.today())
    name = f'{prefix}/{job_args["env"]}/{job_args["rl"]}'
    if args.sweep is not None:
        name = '/'.join([name] + [f'{k}_{v}' for k, v in job_args.items()
                                  if k not in ('env', 'rl', 'seed', 'split_type')])
    cmd += f'name={name} '
    cmd += ' '.join([f'{k}={v}' for k, v in job_args.items()])
    cmd += f' cuda_device={gpu}'
    proc = subprocess.Popen(cmd, shell=True)
    RUNNING.append(Job(proc, gpu, job_args))


# Run it!
os.chdir(RLKIT_PROJECT_PATH)
if not os.path.exists(args.job_log_dir):
    os.makedirs(args.job_log_dir)
with open(LOG_PATH, 'w') as f:
    f.write('Timestamp \t Status \t Args\n')
arg_keys = list(ARG_DICT.keys())
num_each_args = np.array([len(ARG_DICT[k]) for k in arg_keys])
arg_idxs = np.array([0 for _ in range(len(ARG_DICT))])
while True:
    add_job({k: ARG_DICT[k][arg_idxs[kidx]] for kidx, k, in enumerate(arg_keys)})
    arg_idxs[0] += 1
    for ii in range(len(arg_idxs) - 1):
        if arg_idxs[ii] >= num_each_args[ii]:
            arg_idxs[ii] = 0
            arg_idxs[ii + 1] += 1
    if np.any(arg_idxs >= num_each_args):
        break
while len(RUNNING) > 0:
    prune_completed_job()
    time.sleep(30)
with open(LOG_PATH, 'w') as f:
    f.write('Done!')
