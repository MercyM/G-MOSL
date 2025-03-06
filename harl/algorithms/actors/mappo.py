"""MAPPO algorithm."""
import numpy as np
import torch
import torch.nn as nn
from harl.utils.envs_tools import check
from harl.utils.models_tools import get_grad_norm
from harl.algorithms.actors.on_policy_base import OnPolicyBase

from torch.autograd import Variable
from harl.utils.mgpo.min_norm_solvers import MinNormSolver, gradient_normalizers
from harl.utils.models_tools import get_grad_norm


class MAPPO(OnPolicyBase):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        """Initialize MAPPO algorithm.
        Args:
            args: (dict) arguments.
            obs_space: (gym.spaces or list) observation space.
            act_space: (gym.spaces) action space.
            device: (torch.device) device to use for tensor operations.
        """
        super(MAPPO, self).__init__(args, obs_space, act_space, device)

        self.clip_param = args["clip_param"]
        self.ppo_epoch = args["ppo_epoch"]
        self.actor_num_mini_batch = args["actor_num_mini_batch"]
        self.entropy_coef = args["entropy_coef"]
        self.use_max_grad_norm = args["use_max_grad_norm"]
        self.max_grad_norm = args["max_grad_norm"]
        self.use_mgda = args["use_mgda"]
        self.use_mgda_epoch = args["use_mgda_epoch"]
        self.mgda_eps = args["mgda_eps"]
        self.chaos_grad = args["chaos_grad"]

    def update(self, sample, epoch=None):
        """Update actor network.
        Args:
            sample: (Tuple) contains data batch with which to update networks.
        Returns:
            policy_loss: (torch.Tensor) actor(policy) loss value.
            dist_entropy: (torch.Tensor) action entropies.
            actor_grad_norm: (torch.Tensor) gradient norm from actor update.
            imp_weights: (torch.Tensor) importance sampling weights.
        """
        (
            obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            active_masks_batch,
            old_action_log_probs_batch,
            adv_targ,
            available_actions_batch,
        ) = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)

        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # reshape to do in a single forward pass for all steps
        action_log_probs, dist_entropy, _ = self.evaluate_actions(
            obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
        )
        # update actor
        imp_weights = getattr(torch, self.action_aggregation)(
            torch.exp(action_log_probs - old_action_log_probs_batch),
            dim=-1,
            keepdim=True,
        )

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self.use_policy_active_masks:
            policy_action_loss = (
                                         -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True) * active_masks_batch
                                 ).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss

        self.actor_optimizer.zero_grad()

        (policy_loss - dist_entropy * self.entropy_coef).backward()

        # actor_model = self.actor
        # Calculate parameters for each component (Critic and GRM for this example)
        # actor_params = self.count_parameters(actor_model)

        # Print the parameters count
        # print(f"Actor parameters: {actor_params}")

        if self.use_mgda:
            if epoch < self.ppo_epoch and self.chaos_grad:
                if self.use_max_grad_norm:
                    actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                else:
                    actor_grad_norm = get_grad_norm(self.actor.parameters())

                self.actor_optimizer.step()
            return policy_loss, dist_entropy, imp_weights, self.actor_optimizer, self.actor.parameters()
        elif self.use_mgda_epoch:
            return policy_loss, dist_entropy, imp_weights, self.actor_optimizer, self.actor.parameters()
        else:
            if self.use_max_grad_norm:
                actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            else:
                actor_grad_norm = get_grad_norm(self.actor.parameters())

            self.actor_optimizer.step()

            return policy_loss, dist_entropy, actor_grad_norm, imp_weights

    def train(self, actor_buffer, advantages, state_type):
        """Perform a training update for non-parameter-sharing MAPPO using minibatch GD.
        Args:
            actor_buffer: (OnPolicyActorBuffer) buffer containing training data related to actor.
            advantages: (np.ndarray) advantages.
            state_type: (str) type of state.
        Returns:
            train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        train_info = {}
        train_info["policy_loss"] = 0
        train_info["dist_entropy"] = 0
        train_info["actor_grad_norm"] = 0
        train_info["ratio"] = 0

        if np.all(actor_buffer.active_masks[:-1] == 0.0):
            return train_info

        if state_type == "EP":
            advantages_copy = advantages.copy()
            advantages_copy[actor_buffer.active_masks[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        if self.use_mgda_epoch:
            policy_grads = {}

        for _ in range(self.ppo_epoch):
            policy_loss_epoch, dist_entropy_epoch, imp_weights_epoch = 0, 0, 0
            if self.use_recurrent_policy:
                data_generator = actor_buffer.recurrent_generator_actor(
                    advantages, self.actor_num_mini_batch, self.data_chunk_length
                )
            elif self.use_naive_recurrent_policy:
                data_generator = actor_buffer.naive_recurrent_generator_actor(
                    advantages, self.actor_num_mini_batch
                )
            else:
                data_generator = actor_buffer.feed_forward_generator_actor(
                    advantages, self.actor_num_mini_batch
                )

            for sample in data_generator:
                if self.use_mgda or self.use_mgda_epoch:
                    policy_loss, dist_entropy, imp_weights, optimizer, parameters = self.update(
                        tuple(sample), _)
                    train_info["policy_loss"] += policy_loss.item()
                    train_info["dist_entropy"] += dist_entropy.item()
                    train_info["actor_grad_norm"] += 0
                    train_info["ratio"] += imp_weights.mean()
                    policy_loss_epoch += policy_loss
                    dist_entropy_epoch += dist_entropy
                    imp_weights_epoch += imp_weights
                else:
                    policy_loss, dist_entropy, actor_grad_norm, imp_weights = self.update(sample)

                    train_info["policy_loss"] += policy_loss.item()
                    train_info["dist_entropy"] += dist_entropy.item()
                    train_info["actor_grad_norm"] += actor_grad_norm
                    train_info["ratio"] += imp_weights.mean()

            if self.use_mgda_epoch:
                policy_grads[_] = []
                for param in self.actor.parameters():
                    if param.grad is not None:
                        policy_grads[_].append(Variable(param.grad.data.clone(), requires_grad=False))

        if self.use_mgda_epoch:
            filter_grad_indx = range(self.ppo_epoch)
            sol, _ = MinNormSolver.find_min_norm_element([policy_grads[t] for t in filter_grad_indx])
            grads = policy_grads[0]
            j = 0
            init = True
            for i in filter_grad_indx:
                for g1, g2 in zip(grads, policy_grads[i]):
                    if init:
                        g1 = sol[j] * g2
                        init = False
                    else:
                        g1 += sol[j] * g2
                j += 1

            i = 0
            for param in self.actor.parameters():
                if param.grad is not None:
                    param.grad = grads[i]
                    i += 1
            assert i == len(grads)
            if self.use_max_grad_norm:
                actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            else:
                actor_grad_norm = get_grad_norm(self.actor.parameters())

            self.actor_optimizer.step()

        num_updates = self.ppo_epoch * self.actor_num_mini_batch

        if self.use_mgda:
            for k in train_info.keys():
                train_info[k] /= num_updates
            return policy_loss_epoch, dist_entropy_epoch, imp_weights_epoch, optimizer, parameters, train_info
        else:
            for k in train_info.keys():
                train_info[k] /= num_updates
            return train_info

    def share_param_train(self, actor_buffer, advantages, num_agents, state_type):
        """Perform a training update for parameter-sharing MAPPO using minibatch GD.
        Args:
            actor_buffer: (list[OnPolicyActorBuffer]) buffer containing training data related to actor.
            advantages: (np.ndarray) advantages.
            num_agents: (int) number of agents.
            state_type: (str) type of state.
        Returns:
            train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        train_info = {}
        train_info["policy_loss"] = 0
        train_info["dist_entropy"] = 0
        train_info["actor_grad_norm"] = 0
        train_info["ratio"] = 0

        if state_type == "EP":
            advantages_ori_list = []
            advantages_copy_list = []
            for agent_id in range(num_agents):
                advantages_ori = advantages.copy()
                advantages_ori_list.append(advantages_ori)
                advantages_copy = advantages.copy()
                advantages_copy[actor_buffer[agent_id].active_masks[:-1] == 0.0] = np.nan
                advantages_copy_list.append(advantages_copy)
            advantages_ori_tensor = np.array(advantages_ori_list)
            advantages_copy_tensor = np.array(advantages_copy_list)
            mean_advantages = np.nanmean(advantages_copy_tensor)
            std_advantages = np.nanstd(advantages_copy_tensor)
            normalized_advantages = (advantages_ori_tensor - mean_advantages) / (
                    std_advantages + 1e-5
            )
            advantages_list = []
            for agent_id in range(num_agents):
                advantages_list.append(normalized_advantages[agent_id])
        elif state_type == "FP":
            advantages_list = []
            for agent_id in range(num_agents):
                advantages_list.append(advantages[:, :, agent_id])

        if self.use_mgda_epoch:
            policy_grads = {}

        for _ in range(self.ppo_epoch):
            policy_loss_epoch, dist_entropy_epoch, imp_weights_epoch = 0, 0, 0
            data_generators = []
            for agent_id in range(num_agents):
                if self.use_recurrent_policy:
                    data_generator = actor_buffer[agent_id].recurrent_generator_actor(
                        advantages_list[agent_id],
                        self.actor_num_mini_batch,
                        self.data_chunk_length,
                    )
                elif self.use_naive_recurrent_policy:
                    data_generator = actor_buffer[agent_id].naive_recurrent_generator_actor(
                        advantages_list[agent_id], self.actor_num_mini_batch
                    )
                else:
                    data_generator = actor_buffer[agent_id].feed_forward_generator_actor(
                        advantages_list[agent_id], self.actor_num_mini_batch
                    )
                data_generators.append(data_generator)

            for n in range(self.actor_num_mini_batch):
                batches = [[] for _ in range(8)]
                for generator in data_generators:
                    sample = next(generator)
                    for i in range(8):
                        batches[i].append(sample[i])
                for i in range(7):
                    batches[i] = np.concatenate(batches[i], axis=0)
                if batches[7][0] is None:
                    batches[7] = None
                else:
                    batches[7] = np.concatenate(batches[7], axis=0)

                if self.use_mgda or self.use_mgda_epoch:
                    policy_loss, dist_entropy, imp_weights, optimizer, parameters = self.update(
                        tuple(batches), _)
                    train_info["policy_loss"] += policy_loss.item()
                    train_info["dist_entropy"] += dist_entropy.item()
                    train_info["actor_grad_norm"] += 0
                    train_info["ratio"] += imp_weights.mean()
                    policy_loss_epoch += policy_loss
                    dist_entropy_epoch += dist_entropy
                    imp_weights_epoch += imp_weights
                else:
                    policy_loss, dist_entropy, actor_grad_norm, imp_weights = self.update(
                        tuple(batches)
                    )

                    train_info["policy_loss"] += policy_loss.item()
                    train_info["dist_entropy"] += dist_entropy.item()
                    train_info["actor_grad_norm"] += actor_grad_norm
                    train_info["ratio"] += imp_weights.mean()

            if self.use_mgda_epoch:
                policy_grads[_] = []
                for param in self.actor.parameters():
                    if param.grad is not None:
                        policy_grads[_].append(Variable(param.grad.data.clone(), requires_grad=False))

        if self.use_mgda_epoch:
            filter_grad_indx = range(self.ppo_epoch)
            sol, _ = MinNormSolver.find_min_norm_element([policy_grads[t] for t in filter_grad_indx])
            grads = policy_grads[0]
            j = 0
            init = True
            for i in filter_grad_indx:
                for g1, g2 in zip(grads, policy_grads[i]):
                    if init:
                        g1 = sol[j] * g2
                        init = False
                    else:
                        g1 += sol[j] * g2
                j += 1

            i = 0
            for param in self.actor.parameters():
                if param.grad is not None:
                    param.grad = grads[i]
                    i += 1
            assert i == len(grads)
            if self.use_max_grad_norm:
                actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            else:
                actor_grad_norm = get_grad_norm(self.actor.parameters())

            self.actor_optimizer.step()

        num_updates = self.ppo_epoch * self.actor_num_mini_batch

        if self.use_mgda:
            for k in train_info.keys():
                train_info[k] /= num_updates
            return policy_loss_epoch, dist_entropy_epoch, imp_weights_epoch, optimizer, parameters, train_info
        else:

            for k in train_info.keys():
                train_info[k] /= num_updates
            return train_info
