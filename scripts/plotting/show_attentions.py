"""
Visualize the attention scheme of the policy.
"""
import argparse
import os
import pickle as pkl

import numpy as np
import matplotlib.pyplot as plt


###########################################################################
# %% Load in arguments.
###########################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--attn_dirs', type=str, required=True)
parser.add_argument('--head_idxs', type=str, required=True)
parser.add_argument('--titles', type=str, required=True)
args = parser.parse_args()
plt.rcParams.update({
    'font.size': 16,
    'legend.fontsize': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
})

###########################################################################
# %% Load in attention masks made.
###########################################################################
all_attns = []
for attn_dir in args.attn_dirs.split(','):
    attns = []
    for attn_pkl in os.listdir(attn_dir):
        with open(os.path.join(attn_dir, attn_pkl), 'rb') as f:
            attns.append(pkl.load(f))
    all_attns.append(attns)

###########################################################################
# %% Accumulate all of the attention masks.
###########################################################################
all_mean_attns = []
for attns in all_attns:
    B, H = attns[0].shape[:2]
    L = np.max([a.shape[-1] for a in attns])
    total = np.zeros((H, L, L))
    counts = np.zeros((H, L, L))
    for attn in attns:
        el = attn.shape[-1]
        total[:, :el, :el] += np.sum(attn, axis=0)
        counts[:, :el, :el] += B
    mean_attn = total / counts
    all_mean_attns.append(mean_attn)


###########################################################################
# %% Show the mean attention.
###########################################################################
head_idxs = [int(hi) for hi in args.head_idxs.split(',')]
titles = args.titles.split(',')
fig, axs = plt.subplots(1, len(head_idxs))
axidx = 0
for head_idx, mean_attn, title in zip(head_idxs, all_mean_attns, titles):
    max_attn = np.max(mean_attn, axis=-1)[..., np.newaxis]
    normd_attn = mean_attn / max_attn
    axs[axidx].imshow(normd_attn[head_idx], cmap='Blues')
    axs[axidx].set_title(title, fontsize=18)
    axs[axidx].set_xlabel('Time Step')
    axidx += 1
axs[0].set_ylabel('Sequence Length')
plt.show()
