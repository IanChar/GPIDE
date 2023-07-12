"""
Plot the performance of tracking problems.

Author: Ian Char
Date: April 19, 2023
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
parser.add_argument('--data_path', type=str, required=True)
parser.add_argument('--env_types', type=str, default='msd,dmsd')
parser.add_argument('--var_types', type=str, default='fixed,small,large')
parser.add_argument('--table_format', type=str, default='grid')
parser.add_argument('--normalize_by_best', action='store_true')
parser.add_argument('--lower_is_noact', action='store_true')
parser.add_argument('--dont_skip_na', action='store_true')
parser.add_argument('--use_std', action='store_true')
parser.add_argument('--full', action='store_true')
parser.add_argument('--prune_nan_rows', action='store_true')
args = parser.parse_args()
NAME_MAP = OrderedDict({
    'msd_rnn_sac': 'GRU',
    'msd_rnn_vib': 'GRU+VIB',
    'msd_transformer_sac': 'Transformer',
    'msd_pid_sac': 'PIDE',
    'msd_gid_sac': 'GPIDE',
    'msd_gidnoattnnosum_sac': 'GPIDE-ES',
    'msd_gidnoattn_sac': 'GPIDE-ESS',
    'msd_gidattn_sac': 'GPIDE-Attn',
})
if args.full:
    EVAL_MAPS = {
        'fixed': ['fixed', 'small', 'large'],
        'small': ['fixed', 'small', 'large'],
        'large': ['fixed', 'small', 'large'],
        'sim': ['sim', 'real'],
        'real': ['sim', 'real'],
    }
else:
    EVAL_MAPS = {
        'fixed': ['fixed', 'large'],
        'small': ['small', 'large'],
        'large': ['large'],
        'sim': ['sim', 'real'],
        'real': ['real'],
    }
HIGH_STEPS = 1000000
LOW_STEPS = HIGH_STEPS * (1 - 0.05)

###########################################################################
# %% Go through and assemble all of the performances.
###########################################################################
env_types = args.env_types.split(',')
var_types = args.var_types.split(',')
perfs = OrderedDict({})
xname = 'exploration/num steps total'
for et in env_types:
    for vt in var_types:
        for tt in EVAL_MAPS[vt]:
            if f'{et}-{vt}' in PID_RESULTS:
                perfs[f'{et} {vt} -> {tt}'] = OrderedDict({
                    'PID Controller': PID_RESULTS[f'{et}-{vt}'][tt],
                })
found_methods = set({})
for data_path in args.data_path.split(','):
    for root, dirs, files in os.walk(data_path):
        for dname in dirs:
            if ('-' in dname
                    and dname.split('-')[0] in env_types
                    and dname.split('-')[1] in var_types):
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
if args.normalize_by_best:
    for et in env_types:
        for tt in var_types:
            worst, best = float('inf'), float('-inf')
            # Go through and find the best and worst.
            for vt in var_types:
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
            for vt in var_types:
                key = f'{et} {vt} -> {tt}'
                if key in perfs:
                    perfs[key] = {kk: (
                        (vv[0] - worst) / (best - worst) * 100,
                        vv[1] / (best - worst) * 100,
                    ) for kk, vv in perfs[key].items()}

###########################################################################
# %% Create the table.
###########################################################################
headers = ['PID Controller'] + [v for k, v in NAME_MAP.items() if k in found_methods]
if args.prune_nan_rows:
    avg_scores = {h: np.mean([v[h][0]
                              for v in perfs.values()
                              if h in v])
                  for h in headers}
else:
    avg_scores = {h: np.mean([v[h][0] if h in v else float('-inf')
                              for v in perfs.values()])
                  for h in headers}
rows = []
for k, v in perfs.items():
    row = [k]
    scores = []
    incomplete_row = False
    for header in headers:
        if header in v:
            score, score_err = v[header]
        else:
            score, score_err = float('-inf'), 0.0
            if args.prune_nan_rows:
                incomplete_row = True
                break
        scores.append(score)
        if 'latex' in args.table_format:
            row.append(f'${score:0.2f}'
                       r' \pm'
                       f'{score_err:0.2f}$')
        else:
            row.append(f'{score:0.2f} +- {score_err:0.2f}')
    if not incomplete_row:
        best_score_idx = np.argmax(scores)
        if 'latex' in args.table_format:
            row[best_score_idx + 1] = r'\boldmath' + row[best_score_idx + 1]
        rows.append(row)
# Come up with the averages.
avg_row = ['Average']
for header in headers:
    avg_row.append(f'{avg_scores[header]:0.2f}')
rows.append(avg_row)
table = tabulate(rows, headers=headers, tablefmt=args.table_format)
print(table)
