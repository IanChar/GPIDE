"""
The General Integral Derivative Encoder.

Author: Ian Char
Date: March 19, 2023
"""
import math
from typing import Callable, Sequence, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

import rlkit.torch.pytorch_util as ptu

RECORD_ATTENTION = False  # BE VERY CAREFUL WITH THIS!!!
ATTENTION_SAVE_DIR = 'attentions/Attn/pyb-anv'

if RECORD_ATTENTION:
    import os
    import pickle as pkl
    os.makedirs(ATTENTION_SAVE_DIR)


def identity(x):
    return x


def empty_encoder(x):
    return ptu.zeros(0)


class GIDEncoder(nn.Module):

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        out_dim: int,
        seq_length: int,
        n_attention_heads: int,
        n_integral_heads: int,
        exp_smoothing_weights: Optional[Sequence[int]],
        embed_dim_per_head: int,
        obs_encode_dim: int,
        act_encode_dim: int,
        rew_encode_dim: int,
        transition_encode_dim: int,
        decoder_hidden_size: int = 0,
        encoder_activation: Callable[[torch.Tensor], torch.Tensor] = identity,
        hidden_activation: Callable[[torch.Tensor], torch.Tensor] = F.relu,
        final_activation: Callable[[torch.Tensor], torch.Tensor] = torch.tanh,
        posn_embedder: nn.Module = identity,
    ):
        """
        Constructor.
        Args:
            obs_dim: Dimension of the observation.
            act_dim: Dimension of the action.
            seq_length: Maximum sequence length that can be accounted for.
            n_attention_heads: Number of heads of attention.
            n_integral_heads: Number of heads that we should sum over.
            exp_smoothing_weights: The smoothing weights for each of the exponential
                smoothing heads. Number of weights corresponds to the number of
                exponential smoothing heads.
            obs_encode_dim: Size of the encoding for the observation.
            act_encode_dim: Size of the encoding for the action.
            rew_encode_dim: Size of the encoding for the reward.
            transition_encode_dim: Size of the encoding for the transition.
            decoder_hidden_size: Number of hidden units in the decoder.
            encoder_activation: The activation to use after applying the observation,
                action, reward, and transition encoding.
            hidden_activation: The activation to use after projecting to hidden layer
                of the decoder. Note that if there are no decoder hidden units. This
                will not be used.
            final_activation: Activation to apply to final output of the module.
            posn_embedder: Positional embedder to be applied to the encoding.
        """
        if exp_smoothing_weights is None:
            exp_smoothing_weights = []
        super().__init__()
        seq_length += 1  # Really need to do + 1 for RL Training.
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.out_dim = out_dim
        self.seq_length = seq_length
        self.embed_dim_per_head = embed_dim_per_head
        self.n_attention_heads = n_attention_heads
        self.n_integral_heads = n_integral_heads
        self.n_expsmoothing_heads = len(exp_smoothing_weights)
        self.n_heads = n_attention_heads + n_integral_heads + self.n_expsmoothing_heads
        self.attn_embedding = n_attention_heads * embed_dim_per_head
        self.integral_embedding = n_integral_heads * embed_dim_per_head
        self.expsmoothing_embedding = self.n_expsmoothing_heads * embed_dim_per_head
        self.total_embedding = self.n_heads * embed_dim_per_head
        self.encoder_activation = encoder_activation
        self.final_activation = final_activation
        self.posn_embedder = posn_embedder
        # Initialize the encoders.
        for encode_name, in_dim, out_dim in (
                ('obs_encoder', obs_dim, obs_encode_dim),
                ('act_encoder', act_dim, act_encode_dim),
                ('rew_encoder', 1, rew_encode_dim),
                ('trn_encoder', obs_dim, transition_encode_dim)):
            setattr(self, encode_name,
                    nn.Linear(in_dim, out_dim) if out_dim > 0
                    else empty_encoder)
        self.total_encode_dim = sum([
            obs_encode_dim,
            act_encode_dim,
            rew_encode_dim,
            transition_encode_dim,
        ])
        # Initialize head projections.
        if self.attn_embedding > 0:
            self.attn_proj = nn.Linear(self.total_encode_dim, 3 * self.attn_embedding,
                                       bias=False)
        else:
            self.attn_proj = None
        if self.integral_embedding > 0:
            self.integral_proj = nn.Linear(self.total_encode_dim,
                                           self.integral_embedding,
                                           bias=False)
        else:
            self.integral_proj = None
        if self.expsmoothing_embedding > 0:
            self.expsmoothing_proj = nn.Linear(self.total_encode_dim,
                                               self.expsmoothing_embedding, bias=False)
        else:
            self.expsmoothing_proj = None
        # Initialize Decoder.
        dhidden = decoder_hidden_size if decoder_hidden_size > 0 else self.out_dim
        self.decoder_fc = nn.Linear(self.total_embedding, dhidden)
        if decoder_hidden_size > 0:
            self.decoder_proj = nn.Linear(dhidden, self.out_dim)
            self.hidden_activation = hidden_activation
        else:
            self.decoder_proj = identity
            self.hidden_activation = identity
        # Initialize normalization layers.
        self.batch_norm = nn.BatchNorm1d(self.total_embedding)
        self.layer_norm = nn.LayerNorm(self.total_encode_dim)
        for bn_name, in_dim, encode_dim in (
                ('obs_bn', obs_dim, obs_encode_dim),
                ('act_bn', act_dim, act_encode_dim),
                ('rew_bn', 1, rew_encode_dim),
                ('trn_bn', obs_dim, transition_encode_dim)):
            setattr(self, bn_name,
                    nn.BatchNorm1d(in_dim) if encode_dim > 0
                    else identity)
        # Initialize the buffers.
        self.register_buffer(
            'causal_mat',
            torch.tril(torch.ones(seq_length, seq_length)).view(1, 1,
                                                                seq_length, seq_length),
        )
        self.register_buffer(
            'integral_attentions',
            torch.tril(torch.ones(seq_length, seq_length))
            .view(1, 1, seq_length, seq_length)
            .repeat(1, self.n_integral_heads, 1, 1)
        )
        alphas = torch.Tensor(exp_smoothing_weights).view(
            1, self.n_expsmoothing_heads, 1, 1)
        exp_vector = torch.cat([
            (1 - alphas) ** t
            for t in range(seq_length - 1, -1, -1)
        ], dim=3)
        exp_mat = alphas * torch.cat([
            torch.cat([
                (exp_vector[:, :, :, -(seq_length - i):] /
                    (1 - (1 - alphas) ** (seq_length - i))),
                torch.zeros(1, self.n_expsmoothing_heads, 1, i),
            ], dim=3) for i in range(seq_length - 1, -1, -1)
        ], dim=2)
        self.register_buffer('exp_mat', exp_mat)
        # Initialization of weights.
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2))

    def forward(self, obs_seq, act_seq, rew_seq, history=None):
        """Forward pass to get encodings.

        Args:
            obs_seq: Observations of shape (batch_size, seq length, obs dim)
            act_seq: Actions of shape (batch size, seq length, act dim)
            rew_seq: Rewards of sequence (batch size, seq length, 1)
            history: TODO Right now this does nothing. We could hypothetically save
                time if we cached the k, q, v and if there is only one layer, but the
                work to benefit trade off does not seem good.

        Returns:
            * Encodings of shape (batch size, seq length, out dim)
            * History, but in this case just None.
        """
        # Tweek the sequences by offsetting by one. Create transition sequence.
        full_obs = torch.cat([
            obs_seq[:, [0]],
            obs_seq,
        ], dim=1)
        obs_seq = full_obs[:, :-1]
        trn_seq = full_obs[:, 1:] - full_obs[:, :-1]
        trn_seq = trn_seq[:, -self.seq_length:]
        obs_seq, act_seq, rew_seq = [seq[:, -self.seq_length:]
                                     for seq in [obs_seq, act_seq, rew_seq]]
        # Create the encoding.
        encoding = self.encoder_activation(torch.cat([
            self.obs_encoder(self.obs_bn(obs_seq.transpose(1, 2)).transpose(1, 2)),
            self.act_encoder(self.act_bn(act_seq.transpose(1, 2)).transpose(1, 2)),
            self.rew_encoder(self.rew_bn(rew_seq.transpose(1, 2)).transpose(1, 2)),
            self.trn_encoder(self.trn_bn(trn_seq.transpose(1, 2)).transpose(1, 2)),
        ], dim=-1))
        # Make projections for each type of head.
        B, L, _ = encoding.size()
        to_cat = []
        if self.attn_proj is not None:
            query, key, attn_projs = [
                vout.view(B, L, self.n_attention_heads, self.embed_dim_per_head)
                    .transpose(1, 2)
                for vout in self.attn_proj(self.layer_norm(
                                           self.posn_embedder(encoding)))
                    .split(self.attn_embedding, dim=2)
            ]
            to_cat.append(attn_projs)
        else:
            query, key = None, None
        if self.integral_proj is not None:
            integral_projs = self.integral_proj(encoding)\
                .view(B, L, self.n_integral_heads, self.embed_dim_per_head)\
                .transpose(1, 2)
            to_cat.append(integral_projs)
        if self.expsmoothing_proj is not None:
            es_projs = self.expsmoothing_proj(encoding)\
                .view(B, L, self.n_expsmoothing_heads, self.embed_dim_per_head)\
                .transpose(1, 2)
            to_cat.append(es_projs)
        all_projs = torch.cat(to_cat, dim=1)
        # Create attention schemes.
        all_attns = []
        if query is not None and key is not None:
            attn = (query @ key.transpose(-2, -1)) * (1.0 / math.sqrt(key.size(-1)))
            attn = attn.masked_fill(self.causal_mat[:, :, :L, :L] == 0, float('-inf'))
            attn = F.softmax(attn, dim=-1)
            all_attns.append(attn)
        if self.integral_proj is not None:
            all_attns.append(self.integral_attentions.repeat(B, 1, 1, 1)[:, :, :L, :L])
        if self.expsmoothing_proj is not None:
            all_attns.append(self.exp_mat.repeat(B, 1, 1, 1)[:, :, :L, :L])
        attn = torch.cat(all_attns, dim=1)
        if RECORD_ATTENTION:
            num_records = len(os.listdir(ATTENTION_SAVE_DIR))
            with open(os.path.join(ATTENTION_SAVE_DIR, f'attn_{num_records}.pkl'),
                      'wb') as f:
                pkl.dump(attn.cpu().numpy(), f)
        # Combine time series values with the attention schemes.
        x = (attn @ all_projs).transpose(1, 2).contiguous().view(B, L,
                                                                 self.total_embedding)
        # Pass these values through the decoder.
        x = self.batch_norm(x.transpose(1, 2)).transpose(1, 2)
        x = self.hidden_activation(self.decoder_fc(x))
        x = self.final_activation(self.decoder_proj(x))
        return x, None, None

    def _init_weights(self, module: torch.nn.Module):
        """Intialize the weights of a module."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
