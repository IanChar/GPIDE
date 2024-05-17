"""
Make grid that shows performance of navigation
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from rlkit.util.pandas import read_stats_into_df
from rlkit.envs.env_constants import PID_RESULTS


###########################################################################
# %% Constants and set up the plotting.
###########################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='traces/long-nav')
parser.add_argument('--xname', type=str, default='exploration/num steps total')
parser.add_argument('--diameter', type=int, default=15)
parser.add_argument('--xlim', type=int, default=int(50 * 1e5))
parser.add_argument('--include_pid', action='store_true')
args = parser.parse_args()
plt.style.use('seaborn')
plt.rcParams.update({
    'font.size': 18,
    'legend.fontsize': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
})
METHODS = [
    'msd_rnn_sac',
    'msd_transformer_sac',
    'msd_pid_sac',
    'msd_gid_sac',
    # 'msd_gidattn_sac',
    # 'msd_gidnoattn_sac',
    # 'msd_gidnoattnnosum_sac',
]
COLOR_MAP = {
    'msd_gid_sac': 'red',
    'msd_pid_sac': 'orange',
    'msd_rnn_sac': 'green',
    'msd_gidattn_sac': 'purple',
    'msd_gidnoattn_sac': 'orange',
    'msd_gidnoattnnosum_sac': 'blue',
    'msd_transformer_sac': 'blue',
}
LEGEND_MAP = {
    'msd_rnn_sac': 'GRU',
    'msd_transformer_sac': 'Transformer',
    'msd_pid_sac': 'PIDE',
    'msd_gid_sac': 'GPIDE',
    'msd_gidattn_sac': 'Attention Only',
    'msd_gidnoattn_sac': 'No Attention',
    'msd_gidnoattnnosum_sac': 'No Attention, No Summation',
}
TYPE_MAP = {
    'nav': {
        'sim': 'No Friction',
        'real': 'Friction',
    }
}
EVAL_MAP = {
    'sim': ['sim', 'real'],
    'real': ['real'],
}
XLIM = [0, args.xlim]

###########################################################################
# %% Go through the list and make all of the plots.
###########################################################################
env_type = 'nav'
var_types = ['sim', 'real']
xname = args.xname
fix, axs = plt.subplots(1, 3)
scores = [[[] for _ in range(len(EVAL_MAP[vt]))] for vt in var_types]
plot_offset = 0
for ridx, vt in enumerate(var_types):
    # Load in all of the methods.
    vt_path = os.path.join(args.data_path, '-'.join([env_type, vt]))
    for cidx, eval_type in enumerate(EVAL_MAP[vt]):
        ax = axs[plot_offset + cidx]
        pid_score = PID_RESULTS[f'{env_type}-{vt}'][eval_type][0]
        if args.include_pid:
            ax.axhline(pid_score, ls='--', color='black', label='PID')
            scores[ridx][cidx].append(pid_score)
    for method in METHODS:
        if method not in COLOR_MAP:
            continue
        try:
            rlkit_df = read_stats_into_df(os.path.join(vt_path, method))
        except FileNotFoundError:
            continue
        for cidx, eval_type in enumerate(EVAL_MAP[vt]):
            ax = axs[plot_offset + cidx]
            yname = f'eval_{eval_type}/Average Returns'
            avg_results = rlkit_df.groupby([xname])[yname].mean().to_numpy()
            err_results = rlkit_df.groupby([xname])[yname].sem().to_numpy()
            xticks = rlkit_df[xname].unique()
            if args.diameter > 0:
                smooth_avg = np.array([
                    np.mean(avg_results[i-args.diameter:i])
                    for i in range(args.diameter, len(avg_results) + 1)])
                smooth_err = np.array([
                    np.mean(err_results[i-args.diameter:i])
                    for i in range(args.diameter, len(err_results) + 1)])
                xticks = xticks[args.diameter - 1:]
            else:
                smooth_avg = avg_results.flatten()
                smooth_err = err_results.flatten()
            label = method.split('_')[1]
            scores[ridx][cidx].append(np.mean(smooth_avg[-500:]))
            ax.plot(xticks, smooth_avg, color=COLOR_MAP[method],
                    label=LEGEND_MAP[method], alpha=0.7)
            ax.fill_between(
                xticks,
                smooth_avg - smooth_err,
                smooth_avg + smooth_err,
                alpha=0.2,
                color=COLOR_MAP[method],
            )
    plot_offset += len(EVAL_MAP[vt])
plot_offset = 0
for ridx, vt in enumerate(var_types):
    for cidx, ct in enumerate(EVAL_MAP[vt]):
        type1 = TYPE_MAP[env_type][vt] if env_type in TYPE_MAP else vt
        type2 = TYPE_MAP[env_type][ct] if env_type in TYPE_MAP else ct
        axs[plot_offset + cidx].set_title(f'{type1}'
                                          r' $\rightarrow$ '
                                          f'{type2}', fontsize=18)
    for cidx in range(len(EVAL_MAP[vt])):
        spread = np.max(scores[ridx][cidx]) - np.min(scores[ridx][cidx])
        ax = axs[plot_offset + cidx]
    plot_offset += len(EVAL_MAP[vt])
for ax in axs.flatten():
    ax.set_xlim(XLIM)
    ax.set_xlabel('Environment Steps')
axs.flatten()[0].legend(loc='upper center', ncol=5, bbox_to_anchor=(1.675, 1.25),
                        fancybox=True, shadow=True)
axs.flatten()[0].set_ylabel('Returns')
plt.show()
