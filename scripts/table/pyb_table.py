"""
Plot the performance on the POMDP locomotion.

Author: Ian Char
Date: March 13, 2023
"""
import argparse
from collections import OrderedDict
import os

import numpy as np
import pandas as pd
from tabulate import tabulate

from rlkit.util.pandas import read_stats_into_df
from rlkit.envs.env_constants import TABLE_RESULTS


###########################################################################
# %% Constants and set up the plotting.
###########################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, required=True)
parser.add_argument('--table_format', type=str, default='grid')
parser.add_argument('--dont_normalize', action='store_true')
parser.add_argument('--use_std', action='store_true')
args = parser.parse_args()

###########################################################################
# %% Constants and set up the plotting.
###########################################################################
ENV_NAMES = [
    'HalfCheetahBLT-P-v0',
    'HopperBLT-P-v0',
    'WalkerBLT-P-v0',
    'AntBLT-P-v0',
    'HalfCheetahBLT-V-v0',
    'HopperBLT-V-v0',
    'WalkerBLT-V-v0',
    'AntBLT-V-v0',
]
ENV_NAME_MAP = {
    'HalfCheetahBLT-P-v0': 'pyb-hcp',
    'HalfCheetahBLT-V-v0': 'pyb-hcv',
    'HopperBLT-P-v0': 'pyb-hpp',
    'HopperBLT-V-v0': 'pyb-hpv',
    'WalkerBLT-P-v0': 'pyb-wkp',
    'WalkerBLT-V-v0': 'pyb-wkv',
    'AntBLT-P-v0': 'pyb-anp',
    'AntBLT-V-v0': 'pyb-anv',
}
MY_METHODS = OrderedDict({
    'pyb_transformer_sac': 'SAC-Transformer',
    # 'pyb_gidnoattnnosum_sac': 'SAC-GID-ES',
    # 'pyb_gidnoattn_sac': 'SAC-GPID',
    # 'pyb_gidattn_sac': 'SAC-GID-Attn',
    'pyb_gid_sac': 'SAC-GPID',
})
BASELINES = OrderedDict({
    'ppo_gru': 'PPO-GRU',
    # 'a2c_gru': 'A2C-GRU',
    # 'sac-lstm-64-oa-separate': 'SAC-LSTM',
    'td3-gru-64-oa-separate': 'TD3-GRU',
    'VRM': 'VRM',
})
HIGH_STEPS = 1500000
LOW_STEPS = HIGH_STEPS * 0.8
NUM_SEEDS = 4

###########################################################################
# %% Get results for methods from this repo.
###########################################################################
my_results = OrderedDict(
    {k: [(float('-inf'), 0.0) for _ in range(len(ENV_NAMES))]
     for k in MY_METHODS.keys()}
)
for en_idx, en in enumerate(ENV_NAMES):
    env_tag = ENV_NAME_MAP[en]
    if not os.path.exists(os.path.join(args.data_path, env_tag)):
        continue
    subdfs = []
    for subdir in os.listdir(os.path.join(args.data_path, env_tag)):
        if 'hydra' not in subdir:
            subdir_path = os.path.join(args.data_path, env_tag, subdir)
            sub_df = read_stats_into_df(subdir_path)
            sub_df.insert(0, 'Run', [subdir for _ in range(len(sub_df))])
            subdfs.append(sub_df)
    rlkit_df = pd.concat(subdfs, ignore_index=True)
    run_names = rlkit_df['Run'].unique()
    run_names.sort()
    rlkit_df = rlkit_df[
        (rlkit_df['exploration/num steps total'] >= LOW_STEPS)
        & (rlkit_df['exploration/num steps total'] <= HIGH_STEPS)]
    scores = rlkit_df.groupby(['Run', 'seed'])['evaluation/Average Returns'].mean()
    scores = scores.to_numpy().reshape(-1, NUM_SEEDS)
    if not args.dont_normalize:
        min_score = TABLE_RESULTS[en]['return']['Markovian']
        high_score = TABLE_RESULTS[en]['return']['Oracle']
        scores = (scores - min_score) / (high_score - min_score) * 100
    mean_scores = np.mean(scores, axis=1)
    err_scores = np.std(scores, axis=1)
    if not args.use_std:
        err_scores /= np.sqrt(NUM_SEEDS)
    for rn, ms, ss in zip(run_names, mean_scores, err_scores):
        if rn in my_results:
            my_results[rn][en_idx] = (ms, ss)

###########################################################################
# %% Get results from the pomdp repo
###########################################################################
baseline_results = OrderedDict(
    {k: [(float('-inf'), 0.0) for _ in range(len(ENV_NAMES))]
     for k in BASELINES.keys()}
)
for en_idx, en in enumerate(ENV_NAMES):
    with open(f'pomdp/{en}_rundown.csv', 'r') as f:
        lines = f.readlines()
    for line in lines:
        name, mean_score, std_score = line.split(',')
        if name in baseline_results:
            mean_score = float(mean_score)
            std_score = float(std_score)
            if not args.dont_normalize:
                min_score = TABLE_RESULTS[en]['return']['Markovian']
                high_score = TABLE_RESULTS[en]['return']['Oracle']
                mean_score = (mean_score - min_score) / (high_score - min_score) * 100
                std_score /= (high_score - min_score)
                std_score *= 100
            if not args.use_std:
                std_score /= np.sqrt(NUM_SEEDS)
            baseline_results[name][en_idx] = (mean_score, std_score)

###########################################################################
# %% Create the table.
###########################################################################
headers = [BASELINES[k] for k in baseline_results.keys()]\
          + [MY_METHODS[k] for k in my_results.keys()]
rows = []
for en_idx, en in enumerate(ENV_NAMES):
    row = [en]
    scores = []
    for br in baseline_results.values():
        scores.append(br[en_idx][0])
        if 'latex' in args.table_format:
            row.append(rf'${br[en_idx][0]:0.2f}'
                       r' \pm '
                       rf'{br[en_idx][1]:0.2f}$')
        else:
            row.append(f'{br[en_idx][0]:0.2f} +- {br[en_idx][1]:0.2f}')
    for mr in my_results.values():
        scores.append(mr[en_idx][0])
        if 'latex' in args.table_format:
            row.append(f'${mr[en_idx][0]:0.2f}'
                       r' \pm '
                       f'{mr[en_idx][1]:0.2f}$')
        else:
            row.append(f'{mr[en_idx][0]:0.2f} +- {mr[en_idx][1]:0.2f}')
    best_score_idx = np.argmax(scores)
    if 'latex' in args.table_format:
        row[best_score_idx + 1] = r'\boldmath' + row[best_score_idx + 1]
    rows.append(row)
# Come up with the averages.
avg_row = ['Average']
for br in baseline_results.values():
    avg_row.append(f'{np.mean([v[0] for v in br]):0.2f}')
for mr in my_results.values():
    avg_row.append(f'{np.mean([v[0] for v in mr]):0.2f}')
rows.append(avg_row)
table = tabulate(rows, headers=headers, tablefmt=args.table_format)
print(table)
