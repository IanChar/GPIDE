"""
Make grid that shows performance and the generalization of each method.
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
parser.add_argument('--data_path', type=str, default='traces/tracking')
parser.add_argument('--env_type', type=str, default='msd')
parser.add_argument('--var_types', type=str, default='fixed,small,large')
parser.add_argument('--xname', type=str, default='exploration/num steps total')
parser.add_argument('--diameter', type=int, default=15)
parser.add_argument('--xlim', type=int, default=int(10 * 1e5))
parser.add_argument('--include_pid', type=int, default=1)
parser.add_argument('--row_plots', action='store_true')
args = parser.parse_args()
plt.style.use('seaborn')
plt.rcParams.update({
    'font.size': 16,
    'legend.fontsize': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
})
COLOR_MAP = {
    'msd_gid_sac': 'red',
    'msd_rnn_sac': 'green',
    'msd_pid_sac': 'orange',
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
    'msd_gidattn_sac': 'GPIDE-ESS',
    'msd_gidnoattn_sac': 'GPIDE-ES',
    'msd_gidnoattnnosum_sac': 'GPIDE-Attn',
}
TYPE_MAP = {
    'nav': {
        'sim': 'No Friction',
        'real': 'Friction',
    }
}
XLIM = [0, args.xlim]

###########################################################################
# %% Go through the list and make all of the plots.
###########################################################################
var_types = args.var_types.split(',')
xname = args.xname
if args.row_plots:
    figs, axs = plt.subplots(1, len(var_types) ** 2)
else:
    figs, axs = plt.subplots(len(var_types), len(var_types))
scores = [[[] for _ in range(len(var_types))] for _ in range(len(var_types))]
for ridx, vt in enumerate(var_types):
    # Load in all of the methods.
    vt_path = os.path.join(args.data_path, '-'.join([args.env_type, vt]))
    for cidx, eval_type in enumerate(var_types):
        if args.row_plots:
            ax = axs[ridx * len(var_types) + cidx]
        else:
            ax = axs[ridx, cidx]
        pid_score = PID_RESULTS[f'{args.env_type}-{vt}'][eval_type][0]
        if args.include_pid:
            ax.axhline(pid_score, ls='--', color='black', label='PID')
            scores[ridx][cidx].append(pid_score)
    for method in os.listdir(vt_path):
        if method not in COLOR_MAP:
            continue
        try:
            rlkit_df = read_stats_into_df(os.path.join(vt_path, method))
        except FileNotFoundError:
            continue
        for cidx, eval_type in enumerate(var_types):
            if args.row_plots:
                ax = axs[ridx * len(var_types) + cidx]
            else:
                ax = axs[ridx, cidx]
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
for ridx, vt in enumerate(var_types):
    if args.row_plots:
        for cidx, ct in enumerate(var_types):
            type1 = TYPE_MAP[args.env_type][vt] if args.env_type in TYPE_MAP else vt
            type2 = TYPE_MAP[args.env_type][ct] if args.env_type in TYPE_MAP else ct
            axs[ridx * len(var_types) + cidx].set_title(f'{type1}'
                                                        r' $\rightarrow$ '
                                                        f'{type2}', fontsize=16)
    else:
        axs[0, ridx].set_title(vt.capitalize(), fontsize=16)
        axs[ridx, 0].set_ylabel(vt.capitalize())
    for cidx in range(len(var_types)):
        spread = np.max(scores[ridx][cidx]) - np.min(scores[ridx][cidx])
        if args.row_plots:
            ax = axs[ridx * len(var_types) + cidx]
        else:
            ax = axs[ridx, cidx]
        ax.set_ylim([
            np.min(scores[ridx][cidx]) - 1.25 * spread,
            np.max(scores[ridx][cidx]) + 0.25 * spread,
        ])
for ax in axs.flatten():
    ax.set_xlim(XLIM)
    ax.set_xlabel('Environment Steps')
# axs.flatten()[0].legend(loc='upper center', ncol=5, bbox_to_anchor=(1.675, 1.6),
#                         fancybox=True, shadow=True)
axs.flatten()[0].legend(loc='upper center', ncol=5, bbox_to_anchor=(1.075, 1.4),
                        fancybox=True, shadow=True)
axs.flatten()[0].set_ylabel('Returns')
plt.show()
