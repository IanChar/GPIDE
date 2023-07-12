"""
Plot the performance on the POMDP locomotion.

Author: Ian Char
Date: March 13, 2023
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from rlkit.util.pandas import read_stats_into_df
from rlkit.envs.env_constants import TABLE_RESULTS

###########################################################################
# %% Constants and set up the plotting.
###########################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default=None)
parser.add_argument('--rlkit_smoothing', type=int, default=100)
parser.add_argument('--baseline_smoothing', type=int, default=10)
parser.add_argument('--hide_baselines', action='store_true')
parser.add_argument('--save_path', type=str)
args = parser.parse_args()

###########################################################################
# %% Constants and set up the plotting.
###########################################################################
ENV_NAMES = [
    ['HalfCheetahBLT-P-v0', 'HopperBLT-P-v0', 'WalkerBLT-P-v0', 'AntBLT-P-v0'],
    ['HalfCheetahBLT-V-v0', 'HopperBLT-V-v0', 'WalkerBLT-V-v0', 'AntBLT-V-v0'],
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
LABEL_MAP = {
    # 'pyb_gid_sac': 'GPIDE',
    'pyb_gidattn_sac': 'GPIDE-Attn',
    'pyb_gidnoattn_sac': 'GPIDE-ESS',
    'pyb_gidnoattnnosum_sac': 'GPIDE-ES',
    # 'pyb_transformer_sac': 'Transformer',
}
# General plotting stuff.
DROPNA = False
XBOUNDS = (0, 1.5 * int(1e6))
COLOR_MAP = {}
DONT_PLOT = ['frame-stack_final', 'sidu_linear']
# neatplot.set_style()
plt.style.use('seaborn')
plt.rcParams.update({
    'font.size': 16,
    'legend.fontsize': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
})
fig, axs = plt.subplots(2, 4)
total_plotted = 0
colors = ['red', 'green', 'purple', 'gold', 'royalblue', 'orange', 'cyan',
          'firebrick', 'gold', 'springgreen', 'darkslateblue', 'crimson', 'palegreen']

###########################################################################
# %% Load in the rlkit info
###########################################################################
if args.data_path is not None:
    for idx in range(8):
        ridx = idx // 4
        cidx = idx % 4
        ax = axs[ridx, cidx]
        subdfs = []
        env_name = ENV_NAME_MAP[ENV_NAMES[ridx][cidx]]
        if not os.path.exists(os.path.join(args.data_path, env_name)):
            continue
        for subdir in os.listdir(os.path.join(args.data_path, env_name)):
            if 'hydra' not in subdir:
                subdir_path = os.path.join(args.data_path, env_name, subdir)
                sub_df = read_stats_into_df(subdir_path)
                sub_df.insert(0, 'Run', [subdir for _ in range(len(sub_df))])
                subdfs.append(sub_df)
        rlkit_df = pd.concat(subdfs, ignore_index=True)
        run_names = rlkit_df['Run'].unique()
        run_names.sort()
        if args.rlkit_smoothing is None:
            diameter = 0
        else:
            diameter = args.rlkit_smoothing
        xname = 'exploration/num steps total'
        yname = 'evaluation/Average Returns'
        avgnp, errnp = [], []
        avg_results = rlkit_df.groupby(['Run', xname])[yname].mean().to_numpy()
        err_results = rlkit_df.groupby(['Run', xname])[yname].sem().to_numpy()
        num_steps = {rn: len(rlkit_df[rlkit_df['Run'] == rn].groupby(['Run', xname])
                             [yname].mean())
                     for rn in run_names}
        curri = 0
        for name in run_names:
            ns = num_steps[name]
            avgnp.append(avg_results[curri:curri+ns])
            errnp.append(err_results[curri:curri+ns])
            curri += ns
        if diameter > 0:
            smooth_avg = [np.array([np.mean(avgnp[j][i-diameter:i])
                                    for i in range(diameter, len(avgnp[j]) + 1)])
                          for j in range(len(avgnp))]
            smooth_err = [np.array([np.mean(errnp[j][i-diameter:i])
                                    for i in range(diameter, len(errnp[j]) + 1)])
                          for j in range(len(errnp))]
        else:
            smooth_avg = avgnp
            smooth_err = errnp
        for ridx, rn in enumerate(run_names):
            if rn not in LABEL_MAP:
                continue
            xticks = rlkit_df[rlkit_df['Run'] == rn][xname].unique()
            strt = diameter - 1
            endpt = len(xticks)
            if rn in COLOR_MAP:
                color = COLOR_MAP[rn]
            else:
                color = colors[total_plotted]
                COLOR_MAP[rn] = color
                total_plotted += 1
            ax.plot(xticks[strt:endpt], smooth_avg[ridx],
                    color=color, label=LABEL_MAP[rn], alpha=0.8)
            ax.fill_between(
                xticks[strt:endpt],
                smooth_avg[ridx] - smooth_err[ridx],
                smooth_avg[ridx] + smooth_err[ridx],
                alpha=0.2,
                color=color,
            )

###########################################################################
# %% Load in pomdp saved data.
###########################################################################
# LABEL_DICT = {
#     'ours_sac': 'SAC-LSTM',
#     'ours_td3': 'TD3-GRU',
#     'ppo_gru_ppo': 'PPO-GRU',
#     'a2c_gru_a2c': 'A2C-GRU',
#     'VRM_sac': 'VRM',
# }
# if args.baseline_smoothing is None:
#     diameter = 0
# else:
#     diameter = args.baseline_smoothing
# if not args.hide_baselines:
#     for idx in range(8):
#         ridx = idx // 4
#         cidx = idx % 4
#         ax = axs[ridx, cidx]
#         subdfs = []
#         env_name = ENV_NAMES[ridx][cidx]
#         pomdp_df = pd.read_csv(f'pomdp/{env_name}.csv')
#         baselines = [
#             ('ours', 'sac', 'oar', 'lstm', 64, 'separate'),
#             ('ours', 'td3', 'oa', 'gru', 64, 'separate'),
#             ('ppo_gru', 'ppo', 'o', 'gru', 128, 'shared'),
#             ('a2c_gru', 'a2c', 'o', 'gru', 5, 'shared'),
#             ('VRM', 'sac', 'oar', 'lstm', 64, 'separate'),
#         ]
#         for meth, rl, ot, rntype, length, arch in baselines:
#             filtered = pomdp_df[
#                 (pomdp_df['method'] == meth)
#                 & (pomdp_df['RL'] == rl)
#                 & (pomdp_df['Inputs'] == ot)
#                 & (pomdp_df['Encoder'] == rntype)
#                 & (pomdp_df['Len'] == length)
#                 & (pomdp_df['Arch'] == arch)]
#             if len(filtered) == 0:
#                 continue
#             grouped = filtered.groupby(['env_steps'])
#             xticks = grouped['env_steps'].mean().to_numpy()
#             avg = grouped['return'].mean().to_numpy()
#             err = grouped['return'].sem().to_numpy()
#             smoothed_avg = np.array([np.mean(avg[i-diameter:i])
#                                      for i in range(diameter, len(avg) + 1)])
#             smoothed_err = np.array([np.mean(err[i-diameter:i])
#                                      for i in range(diameter, len(err) + 1)])
#             label = LABEL_DICT[f'{meth}_{rl}']
#             if label in COLOR_MAP:
#                 color = COLOR_MAP[label]
#             else:
#                 color = colors[total_plotted]
#                 COLOR_MAP[label] = color
#                 total_plotted += 1
#             ax.plot(xticks[diameter - 1:], smoothed_avg, color=color, label=label)
#             ax.fill_between(
#                 xticks[diameter - 1:],
#                 smoothed_avg - smoothed_err,
#                 smoothed_avg + smoothed_err,
#                 alpha=0.2,
#                 color=color,
#             )

###########################################################################
# %% Show it all
###########################################################################
for idx in range(8):
    ridx = idx // 4
    cidx = idx % 4
    ax = axs[ridx, cidx]
    env_name = ENV_NAMES[ridx][cidx]
    ax.set_title(env_name, fontsize=16)
    ax.set_xlim(XBOUNDS)
    ax.axhline(TABLE_RESULTS[env_name]['return']['Oracle'], ls='--', color='black',
               label='Oracle')
    ax.axhline(TABLE_RESULTS[env_name]['return']['Markovian'], ls=':', color='black',
               label='Markovian')
for ax in axs[1]:
    ax.set_xlabel('Environment Steps')
axs[0, 0].set_ylabel('Returns')
axs[1, 0].set_ylabel('Returns')
axs[0, 1].legend(loc='upper center', ncol=6, bbox_to_anchor=(1.325, 1.35),
                 fancybox=True, shadow=True)
if args.save_path:
    plt.savefig(args.save_path, format='pdf')
else:
    plt.show()
