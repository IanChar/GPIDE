"""
Encode history using an RNN.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

import rlkit.torch.pytorch_util as ptu

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class RNNEncoder(nn.Module):

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        rnn_type: str,
        rnn_hidden_size: int,
        obs_encode_dim: int,
        act_encode_dim: int,
        rew_encode_dim: int,
        rnn_num_layers: int = 1,
        encoder_activation=F.relu,
        **kwargs
    ):
        super().__init__()
        if obs_encode_dim <= 0:
            raise ValueError('Require obs encoder to be positive integer.')
        self.obs_encoder = nn.Linear(obs_dim, obs_encode_dim)
        if act_encode_dim > 0:
            self.act_encoder = nn.Linear(act_dim, act_encode_dim)
        else:
            self.act_encoder = None
        if rew_encode_dim > 0:
            self.rew_encoder = nn.Linear(1, rew_encode_dim)
        else:
            self.rew_encoder = None
        total_encode_dim = obs_encode_dim + act_encode_dim + rew_encode_dim
        if rnn_type.lower() == 'gru':
            rnn_class = torch.nn.GRU
        elif rnn_type.lower() == 'lstm':
            rnn_class = torch.nn.LSTM
        else:
            raise ValueError(f'Cannot recognize RNN type {rnn_type}')
        self._memory_unit = rnn_class(total_encode_dim, rnn_hidden_size,
                                      num_layers=rnn_num_layers,
                                      batch_first=True)
        self.out_dim = rnn_hidden_size
        self.encoder_activation = encoder_activation
        # default gru initialization is uniform, not recommended
        # https://smerity.com/articles/2016/orthogonal_init.html
        # orthogonal has eigenvalue = 1 to prevent grad explosion or vanishing
        # Taken from Recurrent Networks are Strong Baseline for POMDP
        for name, param in self._memory_unit.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param)

    def forward(self, obs_seq, act_seq, rew_seq, history=None):
        """Forward pass to get encodings.

        Args:
            obs_seq: Observations of shape (batch_size, seq length, obs dim)
            act_seq: Actions of shape (batch size, seq length, act dim)
            rew_seq: Rewards of sequence (batch size, seq length, 1)
            history: Previous encoding if this is here then we do not have to
                re-encode the full previous sequence and just need to take the last
                element of the sequence of each thing.
                If GRU expected shape is
                    (rnn_num_layers, batch_size, rnn_hidden_size)
                If LSTM expected tuple of two of the above.

        Returns:
            * Encodings of shape (batch size, seq length, out dim)
            * History of shape (num_layers, batch_size, rnn_hidden_size) and a tuple
              of these if it is an LSTM.
        """
        if history is not None:
            obs_seq, act_seq, rew_seq = [seq[:, [-1]]
                                         if seq is not None
                                         else None
                                         for seq in (obs_seq, act_seq, rew_seq)]
        obs_encoding = self.encoder_activation(self.obs_encoder(obs_seq))
        encoding = [obs_encoding]
        if self.act_encoder is not None:
            encoding.append(self.encoder_activation(self.act_encoder(act_seq)))
        if self.rew_encoder is not None:
            encoding.append(self.encoder_activation(self.rew_encoder(rew_seq)))
        encoding = torch.cat(encoding, dim=-1)
        encoding, new_history = self._memory_unit(encoding, history)
        return encoding, new_history, None


class VariationalRNNEncoder(RNNEncoder):

    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.mean_decoder = nn.Linear(kwargs['rnn_hidden_size'],
                                      kwargs['rnn_hidden_size'])
        self.logvar_decoder = nn.Linear(kwargs['rnn_hidden_size'],
                                        kwargs['rnn_hidden_size'])

    def forward(self, obs_seq, act_seq, rew_seq, history=None):
        """Forward pass to get encodings.

        Args:
            obs_seq: Observations of shape (batch_size, seq length, obs dim)
            act_seq: Actions of shape (batch size, seq length, act dim)
            rew_seq: Rewards of sequence (batch size, seq length, 1)
            history: Previous encoding if this is here then we do not have to
                re-encode the full previous sequence and just need to take the last
                element of the sequence of each thing.
                If GRU expected shape is
                    (rnn_num_layers, batch_size, rnn_hidden_size)
                If LSTM expected tuple of two of the above.

        Returns:
            * Encodings of shape (batch size, seq length, out dim)
            * History of shape (num_layers, batch_size, rnn_hidden_size) and a tuple
              of these if it is an LSTM.
        """
        encoding, new_history, _ = super().forward(obs_seq, act_seq, rew_seq, history)
        means = self.mean_decoder(encoding)
        logvars = self.logvar_decoder(encoding)
        logvars = torch.clamp(
            logvars,
            2 * LOG_SIG_MIN,
            2 * LOG_SIG_MAX
        )
        # Compute KL divergence with prior (isotropic standard gaussian).
        # This should be sum instead of mean but let's put mean so that it is
        # relatively the same scale regardless of number of latent variables.
        # kldivs should be shape (B, L, 1)
        kldivs = -0.5 * (1 + logvars
                         - means.pow(2)
                         - logvars.exp()).mean(dim=-1, keepdim=True)
        if self.training:
            latent = (torch.normal(0, 1, size=means.shape, device=ptu.device)
                      * (0.5 * logvars).exp() + means)
        else:
            latent = means
        return latent, new_history, kldivs
