from collections import OrderedDict
from typing import Tuple

import numpy as np
import torch
from rlkit.core.loss import LossStatistics

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.sac.sac import SACTrainer, SACLosses


class SequentialSACTrainer(SACTrainer):

    def compute_loss(
        self,
        batch,
        skip_statistics=False,
    ) -> Tuple[SACLosses, LossStatistics]:
        """Compute the loss.

        Args:
            batch: For this version we expect the batch to have.
                observations: (batch_len, L, obs_dim)
                next_observations: (batch_len, L, obs_dim)
                actions: (batch_len, L + 1, act_dim)
                terminals: (batch_len, L, 1)
                masks: (batch_len, L, 1)
                rewards: (batch_len, L + 1, 1)
        """
        terminals = batch['terminals']
        obs = batch['observations']
        next_obs = batch['next_observations']
        masks = batch['masks']
        actions = batch['actions'][:, 1:]
        prev_acts = batch['actions'][:, :-1]
        rews = batch['rewards'][:, 1:]
        prev_rews = batch['rewards'][:, :-1]
        full_obs = torch.cat([obs, next_obs[:, [-1]]], dim=1)
        full_acts = batch['actions']
        full_rews = batch['rewards']
        num_valid = masks.sum()
        """
        Policy and Alpha Loss
        """
        new_obs_actions, log_pi, pi_means, pi_stds, pi_kls =\
            self.policy(obs, prev_acts, prev_rews)[:5]
        if self.use_automatic_entropy_tuning:
            current_log_probs = ((log_pi * masks).sum() / num_valid).item()
            alpha_loss = -(self.log_alpha.exp()
                           * (current_log_probs + self.target_entropy))
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = 1
        q_new_actions = torch.min(
            self.qf1(obs, prev_acts, prev_rews, new_obs_actions)[0],
            self.qf2(obs, prev_acts, prev_rews, new_obs_actions)[0],
        )
        policy_loss = ((alpha*log_pi - q_new_actions) * masks).sum() / num_valid
        if self.policy_kl_beta > 0.0:
            if pi_kls is None:
                raise ValueError('Incompatible arch for kl_beta > 0')
            policy_loss += self.policy_kl_beta * (pi_kls * masks).sum() / num_valid

        """
        QF Loss
        """
        q1_pred = self.qf1(obs, prev_acts, prev_rews, actions)[0] * masks
        q2_pred = self.qf2(obs, prev_acts, prev_rews, actions)[0] * masks
        new_next_actions, new_log_pi = self.policy(
            full_obs, full_acts, full_rews)[:2]
        # Note that we have to trim off first step because we do not care about
        # it for next values.
        target_q_values = torch.min(
            self.target_qf1(full_obs, full_acts, full_rews, new_next_actions)[0],
            self.target_qf2(full_obs, full_acts, full_rews, new_next_actions)[0],
        )[:, 1:] - alpha * new_log_pi[:, 1:]
        q_target = (self.reward_scale * rews
                    + (1. - terminals) * self.discount * target_q_values)
        q_target = q_target.detach()
        if self.max_value is not None:
            q_target = torch.clamp(q_target, max=self.max_value)
        q_target *= masks
        qf1_loss = ((q1_pred - q_target) ** 2).sum() / num_valid
        qf2_loss = ((q2_pred - q_target) ** 2).sum() / num_valid

        """
        Save some statistics for eval
        """
        eval_statistics = OrderedDict()
        if not skip_statistics:
            eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'policy/mean',
                ptu.get_numpy(pi_means),
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'policy/normal/std',
                ptu.get_numpy(pi_stds),
            ))
            if pi_kls is not None:
                eval_statistics.update(create_stats_ordered_dict(
                    'policy/latent_KL',
                    ptu.get_numpy(pi_kls),
                ))
            if self.use_automatic_entropy_tuning:
                eval_statistics['Alpha'] = alpha.item()
                eval_statistics['Alpha Loss'] = alpha_loss.item()
        loss = SACLosses(
            policy_loss=policy_loss,
            qf1_loss=qf1_loss,
            qf2_loss=qf2_loss,
            alpha_loss=alpha_loss,
        )
        return loss, eval_statistics
