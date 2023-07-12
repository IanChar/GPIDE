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
parser.add_argument('--attn_dir', type=str, default='temp/attention')
args = parser.parse_args()

###########################################################################
# %% Load in attention masks made.
###########################################################################
attns = []
for attn_pkl in os.listdir(args.attn_dir):
    with open(os.path.join(args.attn_dir, attn_pkl), 'rb') as f:
        attns.append(pkl.load(f))

###########################################################################
# %% Accumulate all of the attention masks.
###########################################################################
padded_attns = []
B, H = attns[0].shape[:2]
L = np.max([a.shape[-1] for a in attns])
total = np.zeros((H, L, L))
counts = np.zeros((H, L, L))
for attn in attns:
    el = attn.shape[-1]
    total[:, :el, :el] += np.sum(attn, axis=0)
    counts[:, :el, :el] += B
mean_attn = total / counts

###########################################################################
# %% Show the mean attention.
###########################################################################
max_attn = np.max(mean_attn, axis=-1)[..., np.newaxis]
normd_attn = mean_attn / max_attn
plt.rcParams.update({
    'font.size': 16,
    'legend.fontsize': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
})
fig, axs = plt.subplots(1, H)
for h in range(H):
    axs[h].imshow(normd_attn[h], cmap='Blues')
    axs[h].set_xlabel('Time Step')
axs[0].set_ylabel('Sequence Length')
plt.show()
