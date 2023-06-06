"""
Policy that will output tanh Gaussians.

Author: Ian Char
Date: Feburary 10, 2023
"""
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from rlkit.torch.core import torch_ify
import rlkit.torch.pytorch_util as ptu
from rlkit.torch.networks import Mlp

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class SequentialGaussianPolicy(Mlp):

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        decoder_hidden_sizes: Optional[Sequence[int]],
        encoder,  # A SequenceEncoder
        obs_encode_dim: int,
        std=None,
        init_w: float = 1e-3,
        hidden_activation=F.relu,
        output_size=None,
        **kwargs
    ):
        if output_size is None:
            output_size = action_dim
        if decoder_hidden_sizes is None:
            decoder_hidden_sizes = []
        super().__init__(
            decoder_hidden_sizes,
            input_size=encoder.out_dim + obs_encode_dim,
            output_size=output_size,
            init_w=init_w,
            hidden_activation=hidden_activation,
            **kwargs
        )
        self.action_dim = action_dim
        self.encoder = encoder
        self.log_std = None
        self.std = std
        self.deterministic = False
        self.encode_history = None  # Previous encodings made.
        self.last_actions = None  # Previous actions played.
        self.last_rewards = None  # Previous rewards observed.
        self.obs_encoder = nn.Linear(obs_dim, obs_encode_dim)
        if std is None:
            last_hidden_size = encoder.out_dim + obs_encode_dim
            if len(decoder_hidden_sizes) > 0:
                last_hidden_size = decoder_hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, output_size)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def reset(self):
        self.encode_history = None
        self.obs_history = None
        self.act_history = None
        self.rew_history = None

    def forward(self, obs_seq, act_seq, rew_seq, history=None):
        """
        Forward pass. Where B = batch_size, L = seq length..
            * obs_seq: (B, L, obs_dim)
            * act_seq: (B, L, act_dim)
            * rew_seq: (B, L, 1)

        Outputs:
            * actions (B, L, act_dim)
            * means: (B, L, act_dim)
            * stds: (B, L, act_dim)
            * kldivs: (B, L, 1)
            * history: Depends on the sequence encoder.
        """
        mean, std, kldivs, history = self._get_net_outs(
            obs_seq, act_seq, rew_seq, history)
        normal_sample = (torch.normal(0, 1, size=mean.shape, device=ptu.device)
                         * std + mean)
        actions = torch.tanh(normal_sample)
        logprobs = torch.sum(
            -0.5 * ((normal_sample - mean) / std).pow(2)
            - torch.log(std)
            - 0.5 * ptu.from_numpy(np.log([2.0 * np.pi])),
            dim=-1,
            keepdim=True,
        )
        logprobs -= 2.0 * (
            ptu.from_numpy(np.log([2.]))
            - normal_sample
            - torch.nn.functional.softplus(-2.0 * normal_sample)
        ).sum(dim=-1, keepdim=True)
        return actions, logprobs, mean, std, kldivs, history

    def get_action(self, obs_np, ):
        """Get action.

        Args:
            obs_np: numpy array of shape (obs_dim,)
        """
        actions, logprobs = self.get_actions(obs_np[None])
        return actions[0, :], {'logpi': float(logprobs)}

    def get_actions(self, obs_np, ):
        """Get multiple actions.

        Args:
            obs_np: numpy array of shape (batch_size, obs_dim)
        """
        # Do forward pass.
        obs_seq = torch_ify(obs_np).unsqueeze(1)
        if self.obs_history is None:
            self.obs_history = obs_seq
            self.act_history = ptu.zeros(len(obs_seq), 1, self.action_dim)
            self.rew_history = ptu.zeros(len(obs_seq), 1, 1)
        else:
            self.obs_history = torch.cat([self.obs_history, obs_seq], dim=1)
        with torch.no_grad():
            actions, logprobs, mean, std, _, history =\
                self.forward(self.obs_history, self.act_history,
                             self.rew_history,
                             history=self.encode_history)
        actions, logprobs, mean, std = [seq[:, -1] for seq in
                                        (actions, logprobs, mean, std)]
        self.encode_history = history
        # Sample actions.
        if self.deterministic:
            actions = torch.tanh(mean)
            logprobs = ptu.ones(len(actions))
        self.act_history = torch.cat([self.act_history, actions.unsqueeze(1)],
                                     dim=1)
        return ptu.get_numpy(actions), ptu.get_numpy(logprobs.squeeze(-1))

    def get_reward_feedback(self, rewards):
        if isinstance(rewards, float):
            rewards = np.array([rewards])
        rewards = ptu.from_numpy(rewards).view(-1, 1, 1)
        if self.rew_history is None:
            self.rew_history = rewards
        else:
            self.rew_history = torch.cat([self.rew_history, rewards], dim=1)

    def _get_net_outs(self, obs_seq, act_seq, rew_seq, history=None):
        """
        Forward pass. Where B = batch_size, L = seq length..
            * obs_seq: (B, L, obs_dim)
            * act_seq: (B, L, act_dim)
            * rew_seq: (B, L, 1)

        Outputs:
            * means: (B, L, act_dim)
            * stds: (B, L, act_dim)
            * kldivs: (B, L, 1)
            * history: Depends on the sequence encoder.
        """
        encoding, history, encoder_kl = self.encoder(obs_seq, act_seq, rew_seq, history)
        obs_encoding = self.hidden_activation(self.obs_encoder(
            obs_seq[:, -encoding.shape[1]:]))
        if encoding.shape[1] == 1:
            obs_encoding = obs_encoding[:, [-1]]
        h = torch.cat([encoding, obs_encoding], dim=-1)
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = torch.from_numpy(np.array([self.std, ])).float().to(
                ptu.device)
        return mean, std, encoder_kl, history


class VIBSequentialGaussianPolicy(SequentialGaussianPolicy):

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int,
        latent_encoder_hidden_sizes: Sequence[int],
        decoder_hidden_sizes: Sequence[int],
        encoder,  # A SequenceEncoder
        obs_encode_dim: int,
        std=None,
        init_w: float = 1e-3,
        hidden_activation=F.relu,
        output_size=None,
        **kwargs
    ):
        # Super constructor will construct the encoding for the latent variable.
        super().__init__(
            obs_dim=obs_dim,
            action_dim=action_dim,
            decoder_hidden_sizes=latent_encoder_hidden_sizes,
            encoder=encoder,
            obs_encode_dim=obs_encode_dim,
            std=None,
            init_w=init_w,
            hidden_activation=hidden_activation,
            output_size=latent_dim,
            **kwargs
        )
        # Now make the actual decoder that takes in latent draws.
        if output_size is None:
            output_size = action_dim
        if decoder_hidden_sizes is None:
            decoder_hidden_sizes = []
        self.decoder = Mlp(
            hidden_sizes=decoder_hidden_sizes,
            output_size=output_size,
            input_size=latent_dim,
            init_w=init_w,
            hidden_activation=hidden_activation,
            **kwargs
        )
        if std is None:
            last_hidden_size = encoder.out_dim,
            if len(decoder_hidden_sizes) > 0:
                last_hidden_size = decoder_hidden_sizes[-1]
            self.decoder_log_std = nn.Linear(last_hidden_size, action_dim)
            self.decoder_log_std.weight.data.uniform_(-init_w, init_w)
            self.decoder_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.decoder_log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def _get_net_outs(self, obs_seq, act_seq, rew_seq, history=None):
        """
        Forward pass. Where B = batch_size, L = seq length..
            * obs_seq: (B, L, obs_dim)
            * act_seq: (B, L, act_dim)
            * rew_seq: (B, L, 1)

        Outputs:
            * means: (B, L, act_dim)
            * stds: (B, L, act_dim)
            * kldiv: (B, L, 1)
            * history: Depends on the sequence encoder.
        """
        # Make the encoding.
        encoding, history, encoder_kl = self.encoder(obs_seq, act_seq, rew_seq, history)
        obs_encoding = self.hidden_activation(self.obs_encoder(obs_seq))
        if encoding.shape[1] == 1:
            obs_encoding = obs_encoding[:, [-1]]
        h = torch.cat([encoding, obs_encoding], dim=-1)
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        latent_mean = self.last_fc(h)
        latent_log_var = torch.clamp(
            self.last_fc_log_std(h),
            2 * LOG_SIG_MIN,
            2 * LOG_SIG_MAX,
        )
        # Compute KL divergence with prior (isotropic standard gaussian).
        # This should be sum instead of mean but let's put mean so that it is
        # relatively the same scale regardless of number of latent variables.
        kldivs = -0.5 * (1 + latent_log_var
                         - latent_mean.pow(2)
                         - latent_log_var.exp()).mean(dim=-1, keepdim=True)
        if encoder_kl is not None:
            kldivs += encoder_kl
        # Sample from the latent.
        if self.deterministic:
            latent = latent_mean
        else:
            latent = (torch.normal(0, 1, size=latent_mean.shape, device=ptu.device)
                      * (0.5 * latent_log_var).exp() + latent_mean)
        # Decode into actions.
        h = latent
        for i, fc in enumerate(self.decoder.fcs):
            h = fc(h)
            h = self.hidden_activation(h)
        mean = self.decoder.last_fc(h)
        if self.std is None:
            log_std = self.decoder_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = torch.from_numpy(np.array([self.std, ])).float().to(
                ptu.device)
        return mean, std, kldivs, history
