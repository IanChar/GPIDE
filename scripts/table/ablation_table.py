"""
Make a table comparing the average performances between versions of GPID.

Author: Ian Char
Date: May 10, 2023
"""
import argparse
from collections import OrderedDict
import os

import numpy as np
from tabulate import tabulate

from rlkit.util.pandas import read_stats_into_df
from rlkit.envs.env_constants import PID_RESULTS, NO_ACTION_RESULTS


###########################################################################
# %% Constants and set up the plotting.
###########################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='logs/final/pomdp')
parser.add_argument('--env_types', type=str, default='msd,dmsd,nav,fusion,bnrot')
parser.add_argument('--var_types', type=str, default='fixed,small,large')
parser.add_argument('--table_format', type=str, default='grid')
parser.add_argument('--normalize_by_winner', action='store_true')
parser.add_argument('--normalize_by_best', action='store_true')
parser.add_argument('--lower_is_noact', action='store_true')
parser.add_argument('--dont_skip_na', action='store_true')
parser.add_argument('--use_std', action='store_true')
args = parser.parse_args()
NAME_MAP = OrderedDict({
    'msd_gidnoattnnosum_sac': 'ExpSmoothing',
    'msd_gidnoattn_sac': 'ExpSmoothing + Summation',
    'msd_gidattn_sac': 'Attention',
    'msd_gid_sac': 'GPID',
    'msd_rnn_sac': 'GRU',
    'msd_transformer_sac': 'Transformer',
    'msd_pid_sac': 'PID',
})
VAR_TYPE_MAP = {
    'msd': ['fixed', 'small', 'large'],
    'dmsd': ['fixed', 'small', 'large'],
    'nav': ['sim', 'real'],
    'fusion': ['sim', 'real'],
    'bnrot': ['sim', 'real'],
}
EVAL_MAPS = {
    'fixed': ['fixed', 'large'],
    'small': ['small', 'large'],
    'large': ['large'],
    'sim': ['sim', 'real'],
    'real': ['real'],
}
HEADER_MAP = {
    'msd': 'MSD',
    'dmsd': 'DMSD',
    'nav': 'Navigation',
    'fusion': 'Bn Track',
    'bnrot': 'BnRot Track',
}
HIGH_STEPS = 1000000
LOW_STEPS = HIGH_STEPS * (1 - 0.05)

###########################################################################
# %% Go through and assemble all of the performances.
###########################################################################
env_types = args.env_types.split(',')
perfs = OrderedDict({})
xname = 'exploration/num steps total'
for et in env_types:
    for vt in VAR_TYPE_MAP[et]:
        for tt in EVAL_MAPS[vt]:
            perfs[f'{et} {vt} -> {tt}'] = OrderedDict({
                'PID Controller': PID_RESULTS[f'{et}-{vt}'][tt],
            })
found_methods = set({})
for data_path in args.data_path.split(','):
    for root, dirs, files in os.walk(data_path):
        for dname in dirs:
            if ('-' in dname
                    and dname.split('-')[0] in env_types
                    and dname.split('-')[1] in VAR_TYPE_MAP[dname.split('-')[0]]):
                et, vt = dname.split('-')[:2]
                vt_path = os.path.join(root, dname)
                for method in os.listdir(vt_path):
                    if method not in NAME_MAP:
                        continue
                    try:
                        rlkit_df = read_stats_into_df(os.path.join(vt_path, method))
                    except FileNotFoundError:
                        continue
                    found_methods.add(method)
                    rlkit_df = rlkit_df[
                        (rlkit_df[xname] >= LOW_STEPS)
                        & (rlkit_df[xname] <= HIGH_STEPS)]
                    for tt in EVAL_MAPS[vt]:
                        yname = f'eval_{tt}/Average Returns'
                        if not args.dont_skip_na:
                            rlkit_df.fillna(method='ffill', inplace=True)
                        scores = rlkit_df.groupby(['seed'])[yname].mean().to_numpy()
                        num_seeds = len(scores)
                        perfs[f'{et} {vt} -> {tt}'][NAME_MAP[method]] = (
                            np.mean(scores),
                            np.std(scores) / np.sqrt(num_seeds),
                        )

###########################################################################
# %% Do normalizations.
###########################################################################
if args.normalize_by_winner:
    for k, v in perfs.items():
        mean_scores = [vv[0] for vv in v.values()]
        worst, best = np.min(mean_scores), np.max(mean_scores)
        if args.lower_is_noact:
            et, _, _, tt = k.split(' ')
            worst = NO_ACTION_RESULTS[et][tt][0]
        perfs[k] = {kk: (
            (vv[0] - worst) / (best - worst) * 100,
            vv[1] / (best - worst) * 100,
        ) for kk, vv in v.items()}
elif args.normalize_by_best:
    for et in env_types:
        for tt in VAR_TYPE_MAP[et]:
            worst, best = float('inf'), float('-inf')
            # Go through and find the best and worst.
            for vt in VAR_TYPE_MAP[et]:
                key = f'{et} {vt} -> {tt}'
                if key in perfs:
                    mean_scores = [vv[0] for vv in perfs[key].values()]
                    key_worst, key_best = np.min(mean_scores), np.max(mean_scores)
                    if key_worst < worst:
                        worst = key_worst
                    if key_best > best:
                        best = key_best
            if args.lower_is_noact:
                worst = NO_ACTION_RESULTS[et][tt][0]
            # Now go back through and normalize.
            for vt in VAR_TYPE_MAP[et]:
                key = f'{et} {vt} -> {tt}'
                if key in perfs:
                    perfs[key] = {kk: (
                        (vv[0] - worst) / (best - worst) * 100,
                        vv[1] / (best - worst) * 100,
                    ) for kk, vv in perfs[key].items()}

###########################################################################
# %% Create the table.
###########################################################################
headers = ['Ablation'] + [HEADER_MAP[h] for h in env_types]
rows = []
base_scores = []
for et in env_types:
    base_scores.append(np.mean([v[NAME_MAP['msd_gid_sac']][0]
                                for k, v in perfs.items()
                                if k.split(' ')[0] == et]))
for abl in ('msd_gidnoattnnosum_sac', 'msd_gidnoattn_sac', 'msd_gidattn_sac'):
    row = [NAME_MAP[abl]]
    for base_score, et in zip(base_scores, env_types):
        score = np.mean([v[NAME_MAP[abl]][0] for k, v in perfs.items()
                         if k.split(' ')[0] == et])
        row.append(f'{(score - base_score) / base_score * 100:0.2f}%')
    rows.append(row)
table = tabulate(rows, headers=headers, tablefmt=args.table_format)
print(table)
