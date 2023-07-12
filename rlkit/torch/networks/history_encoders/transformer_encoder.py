"""
Encoder that uses a GPT-style transformer.

Much of the code used https://github.com/karpathy/nanoGPT as reference.

Author: Ian Char
Date: April 16, 2023
"""
import math
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    """Causal self attention. This is causal because we only attend on the past.
       Inspired by the code from Karpathy's nanoGPT repo.
    """

    def __init__(
        self,
        embed_dim_per_head: int,
        n_heads: int,
        block_size: int,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        """Constructor.

        Args:
            embed_dim: The dimension of the embedding per head.
            n_heads: The number of heads.
            block_size: Size of the block
            dropout: The amount of dropout.
            bias: Whether to have bias in the linear layer.
        """
        super().__init__()
        self.n_heads = n_heads
        self.embed_dim_per_head = embed_dim_per_head
        self.embed_dim = embed_dim_per_head * n_heads
        # This gives us the key, queries, and values for all heads.
        self.c_proj = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=bias)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.register_buffer(
            'bias',
            torch.tril(torch.ones(block_size, block_size)).view(1, 1,
                                                                block_size, block_size),
        )

    def forward(self, net_in: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            net_in: Input of shape (batch_size, sequence length, embed_dim)

        Returns: Output of size (batch_size, sequence length, embed_dim).
        """
        # B = batch_size, L = sequence length, C = embed_dim
        B, L, C = net_in.size()
        query, key, val = [vout.view(B, L, self.n_heads,
                                     self.embed_dim_per_head).transpose(1, 2)
                           for vout in self.c_proj(net_in).split(self.embed_dim, dim=2)]
        attn = (query @ key.transpose(-2, -1)) * (1.0 / math.sqrt(key.size(-1)))
        attn = attn.masked_fill(self.bias[:, :, :L, :L] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        net_out = (attn @ val).transpose(1, 2).contiguous().view(B, L, C)
        return self.resid_dropout(self.out_proj(net_out))


class GPTBlock(nn.Module):

    def __init__(
        self,
        embed_dim_per_head: int,
        n_heads: int,
        block_size: int,
        dropout: float = 0.0,
        bias: bool = False,
        hidden_activation: Callable[[torch.Tensor], torch.Tensor] = F.relu,
        **kwargs
    ):
        """Constructor.

        Args:
            embed_dim: The dimension of the embedding per head.
            n_heads: The number of heads.
            block_size: Size of the block, i.e. the sequence length.
            dropout: The amount of dropout.
            bias: Whether to have bias in the linear layer.
            c_fc_size: Number of hidden units in output. If 0 then does not do
                any operation after.
        """
        super().__init__()
        embed_dim = embed_dim_per_head * n_heads
        self.attn = CausalSelfAttention(
            embed_dim_per_head=embed_dim_per_head,
            n_heads=n_heads,
            block_size=block_size,
            dropout=dropout,
            bias=bias,
        )
        self.ln1 = torch.nn.LayerNorm(embed_dim)
        self.ln2 = torch.nn.LayerNorm(embed_dim)
        self.c_fc = nn.Linear(embed_dim, 4 * embed_dim, bias=bias)
        self.c_proj = nn.Linear(4 * embed_dim, embed_dim, bias=bias)
        self.mlp_dropout = nn.Dropout(dropout)
        self.hidden_activation = hidden_activation

    def forward(self, net_in: torch.Tensor):
        """Forward pass.

        Args:
            net_in: Network input of shape (batch_size, sequence_length, embed dim)

        Returns: Network output of shape (batch_size, sequence_length, embed_dim)
        """
        net_in = net_in + self.attn(self.ln1(net_in))
        net_in = (net_in
                  + self.mlp_dropout(self.c_proj(
                      self.hidden_activation(self.c_fc(self.ln2(net_in))))))
        return net_in


class SinCosPositionEmbedding(nn.Module):
    """Taken from the pytorch transformers tutorial."""

    def __init__(
        self,
        embed_dim: int,
        dropout: float = 0.0,
        max_len: int = 5000,
    ):
        """Constructor.

        Args:
            embed_dim: The dimension of the embedding.
            dropout: The dropout rate.
            max_len: Maximum length of the sequence.
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float()
                             * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, net_in: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            net_in: Shape of (batch_size, sequence length, embed dim)
        """
        net_in = net_in + self.pe[:, :net_in.size(1)]
        return self.dropout(net_in)


class NoPositionEmbedding(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def identity(x):
    return x


class TransformerEncoder(nn.Module):

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        n_heads: int,
        seq_length: int,
        num_layers: int,
        obs_encode_dim: int,
        act_encode_dim: int,
        rew_encode_dim: int,
        # Encode transitions decides whether we have the token in the block encoder
        #   (s_t-1, a_t-1, r_t-1, s_t - s_t-1)
        #   or (s_t, a_t-1, r_t-1)
        # Depending on what option is selected some offsetting will have to happen.
        # Setting transition_encode_dim to a positive number will turn on the former.
        transition_encode_dim: int = 0,
        posn_embedder: Optional[nn.Module] = None,
        encoder_activation: Callable[[torch.Tensor], torch.Tensor] = identity,
        final_activation: Callable[[torch.Tensor], torch.Tensor] = torch.tanh,
        dropout: float = 0.0,
        bias: bool = True,
        weight_initialization: str = 'tfixup',
        **kwargs
    ):
        super().__init__()
        if posn_embedder is None:
            posn_embedder = NoPositionEmbedding()
        self.posn_embedder = posn_embedder
        self.encoder_activation = encoder_activation
        self.final_activation = final_activation
        self.seq_length = seq_length
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
        if transition_encode_dim > 0:
            self.transition_encoder = nn.Linear(obs_dim, transition_encode_dim)
            self.delta_bn = nn.BatchNorm1d(obs_dim)
        else:
            self.transition_encoders = None
            self.delta_bn = None
        total_encode_dim = (
            obs_encode_dim
            + act_encode_dim
            + rew_encode_dim
            + transition_encode_dim
        )
        self.out_dim = total_encode_dim
        assert total_encode_dim % n_heads == 0
        self.all_blocks = nn.ModuleList([GPTBlock(
            embed_dim_per_head=total_encode_dim // n_heads,
            n_heads=n_heads,
            block_size=seq_length + 1,
            dropout=dropout,
            bias=bias,
            **kwargs
        ) for _ in range(num_layers)])
        # Specialized initialization
        if weight_initialization == 'gpt':
            self.apply(self._gpt_init_weights)
            for pn, p in self.named_parameters():
                if pn.endswith('c_proj.weight'):
                    torch.nn.init.normal_(p, mean=0.0,
                                          std=0.02/math.sqrt(2 * num_layers))
        elif weight_initialization == 'tfixup':
            # This code largely inspired by
            # https://github.com/luckeciano/transformers-metarl
            for p in self.all_blocks.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            for od, embedder in (
                    (obs_encode_dim, self.obs_encoder),
                    (act_encode_dim, self.act_encoder),
                    (rew_encode_dim, self.rew_encoder)):
                if od > 0:
                    for p in embedder.parameters():
                        if p.dim() > 1:
                            torch.nn.init.normal(p, 0, od ** (- 1. / 2.))
                    temp_state_dic = {}
                    for name, param in embedder.named_parameters():
                        if 'weight' in name:
                            temp_state_dic[name] = (
                                ((9 * num_layers) ** (- 1. / 4.)) * param)
                    for name in embedder.state_dict():
                        if name not in temp_state_dic:
                            temp_state_dic[name] = embedder.state_dict()[name]
                    embedder.load_state_dict(temp_state_dic)
            temp_state_dic = {}
            for name, param in self.all_blocks.named_parameters():
                if any(s in name for s in ["c_fc.weight",
                                           "c_proj.weight",
                                           "attn.out_proj.weight"]):
                    temp_state_dic[name] =\
                        (0.67 * (num_layers) ** (- 1. / 4.)) * param
                elif "attn.c_proj.weight" in name:
                    temp_state_dic[name] =\
                        (0.67 * (num_layers) ** (- 1. / 4.)) * (param * (2**0.5))
            for name in self.all_blocks.state_dict():
                if name not in temp_state_dic:
                    temp_state_dic[name] = self.all_blocks.state_dict()[name]
            self.all_blocks.load_state_dict(temp_state_dic)

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
        if self.transition_encoder is not None:
            full_obs = torch.cat([
                obs_seq[:, [0]],
                obs_seq,
            ], dim=1)
            obs_seq = full_obs[:, :-1]
            tran_seq = full_obs[:, 1:] - full_obs[:, :-1]
            tran_seq = tran_seq[:, -(self.seq_length + 1):]
            tran_seq = self.delta_bn(tran_seq.transpose(1, 2)).transpose(1, 2)
        obs_seq, act_seq, rew_seq = [seq[:, -(self.seq_length + 1):]
                                     for seq in [obs_seq, act_seq, rew_seq]]
        obs_encoding = self.encoder_activation(self.obs_encoder(obs_seq))
        encoding = [obs_encoding]
        if self.act_encoder is not None:
            encoding.append(self.encoder_activation(self.act_encoder(act_seq)))
        if self.rew_encoder is not None:
            encoding.append(self.encoder_activation(self.rew_encoder(rew_seq)))
        if self.transition_encoder is not None:
            encoding.append(self.encoder_activation(self.transition_encoder(tran_seq)))
        encoding = torch.cat(encoding, dim=-1)
        encoding = self.posn_embedder(encoding)
        for block in self.all_blocks:
            encoding = block(encoding)
        return self.final_activation(encoding), None, None

    def _gpt_init_weights(self, module: torch.nn.Module):
        """Intialize the weights of a module."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
