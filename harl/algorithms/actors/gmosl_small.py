"""MAPPO algorithm."""
import torch
import torch.nn as nn
import numpy as np
from harl.utils.envs_tools import check
from harl.utils.models_tools import get_grad_norm
from harl.algorithms.actors.on_policy_base import OnPolicyBase

from harl.algorithms.comm.graph_net_trans import Actor_graph
from harl.utils.models_tools import update_linear_schedule
from harl.algorithms.comm.util import *
from harl.models.policy_models.stochastic_graph_policy import StochasticGraphPolicy
from harl.utils.envs_tools import get_shape_from_obs_space, get_shape_from_act_space
from harl.algorithms.comm.util import _h_A
from torch.autograd import Variable
from harl.utils.mgpo.min_norm_solvers import MinNormSolver, gradient_normalizers
import time
from harl.utils.envs_tools import get_shape_from_obs_space


class GMOSL(OnPolicyBase):
    def __init__(self, args, obs_space, act_space, num_agents, agent_id, device=torch.device("cpu")):
        """Initialize MAPPO algorithm.
        Args:
            args: (dict) arguments.
            obs_space: (gym.spaces or list) observation space.
            act_space: (gym.spaces) action space.
            device: (torch.device) device to use for tensor operations.
        """
        super(GMOSL, self).__init__(args, obs_space, act_space, device)

        self.clip_param = args["clip_param"]
        self.ppo_epoch = args["ppo_epoch"]
        self.actor_num_mini_batch = args["actor_num_mini_batch"]
        self.entropy_coef = args["entropy_coef"]
        self.use_max_grad_norm = args["use_max_grad_norm"]
        self.max_grad_norm = args["max_grad_norm"]

        self.lr_graph = args["lr_graph"]
        self.num_agents = num_agents
        self.n_actions = get_shape_from_act_space(act_space)
        self.obs_shape = get_shape_from_obs_space(obs_space)[0]

        self.actor = StochasticGraphPolicy(args, self.obs_space, self.act_space, self.num_agents, self.device)
        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(),
            lr=self.lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )
        self.agent_id = agent_id
        self.args = args

        self.use_mgda = args["use_mgda"]
        self.use_mgda_epoch = args["use_mgda_epoch"]
        self.mgda_eps = args["mgda_eps"]


    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def update(self, sample, all_obs=None, all_actions=None):
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
            last_actions_batch,
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
            all_obs,
            all_actions
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
            policy_action_loss = (-torch.sum(torch.min(surr1, surr2), dim=-1,
                                             keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss - dist_entropy * self.entropy_coef

        self.actor_optimizer.zero_grad()
        policy_loss.backward()

        if self.use_mgda:
            return policy_loss, dist_entropy, imp_weights, self.actor_optimizer
        elif self.use_mgda_epoch:
            return policy_loss, dist_entropy, imp_weights, self.actor_optimizer
        else:
            self.actor_optimizer.step()

            if self.use_max_grad_norm:
                actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            else:
                actor_grad_norm = get_grad_norm(self.actor.parameters())

            return policy_loss, dist_entropy, actor_grad_norm, imp_weights

    def train(self, actor_buffer, advantages, state_type, all_obs=None, all_actions=None):
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
        train_info["loss_graph_actor"] = 0

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
            epoch_grad_list = []

        for _ in range(self.ppo_epoch):
            # Initialize gradient accumulators
            if self.use_mgda_epoch:
                per_epoch_grads = []  # 格式：[param_grad_tensor1, param_grad_tensor2...]

            grad_accumulator = [
                torch.zeros_like(param) for param in self.actor.parameters() if param.requires_grad
            ]
            if self.use_mgda_epoch:
                policy_grads[_] = []

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
                    policy_loss, dist_entropy, imp_weights, optimizer = self.update(
                        sample, all_obs, all_actions)
                    train_info["policy_loss"] += policy_loss.item()
                    train_info["dist_entropy"] += dist_entropy.item()
                    train_info["actor_grad_norm"] += 0
                    train_info["ratio"] += imp_weights.mean()
                    policy_loss_epoch += policy_loss
                    dist_entropy_epoch += dist_entropy
                    imp_weights_epoch += imp_weights

                    # 记录梯度
                    with torch.no_grad():
                        grads = []
                        for param in self.actor.parameters():
                            if param.grad is not None:
                                grad = param.grad.clone()
                                grad = grad / (grad.norm() + 1e-8)  # 归一化
                                grads.append(grad)
                            else:
                                # 如果未计算梯度，填充零张量
                                grads.append(torch.zeros_like(param))
                        per_epoch_grads.append(grads)  # 保存该mini-batch的梯度

                else:
                    policy_loss, dist_entropy, actor_grad_norm, imp_weights = self.update(
                        sample, all_obs, all_actions)

                    train_info["policy_loss"] += policy_loss.item()
                    train_info["dist_entropy"] += dist_entropy.item()
                    train_info["actor_grad_norm"] += actor_grad_norm
                    train_info["ratio"] += imp_weights.mean()

            if self.use_mgda_epoch:

                # policy_grads[_] = [grad.detach() for grad in grad_accumulator]
                # 合并该epoch内所有mini-batch的梯度（平均）
                epoch_avg_grads = [
                    torch.stack([g[i] for g in per_epoch_grads]).mean(dim=0)
                    for i in range(len(per_epoch_grads[0]))
                ]
                epoch_grad_list.append(epoch_avg_grads)  # 保存该epoch的梯度

                # for param in self.actor.parameters():
                #     if param.grad is not None:
                #         policy_grads[_].append(Variable(param.grad.data.clone(), requires_grad=False))

        #########################################多目标优化################################################################
        if self.use_mgda_epoch:
            if len(epoch_grad_list) > 0:
                gradients = epoch_grad_list

                # 2. 使用最小范数求解帕累托最优权重
                sol, min_norm = MinNormSolver.find_min_norm_element_FW(gradients)

                # 3. 按权重合并梯度
                combined_grad = [
                    sum(sol[i] * gradients[i][p]
                        for i in range(len(gradients)))
                    for p in range(len(gradients[0]))
                ]

                # 4. 获取模型参数并检查形状
                params = list(self.actor.parameters())
                if len(params) != len(combined_grad):
                    raise RuntimeError(
                        f"参数数量不匹配：模型有 {len(params)} 个参数，但梯度有 {len(combined_grad)} 个"
                    )

                # 5. 调试：检查每个参数和梯度的形状
                for param_idx, (param, grad) in enumerate(zip(params, combined_grad)):
                    if param.size() != grad.size():
                        raise RuntimeError(
                            f"梯度形状不匹配：参数 {param_idx} 的形状为 {param.size()}，梯度形状为 {grad.size()}\n"
                            f"参数：{param}\n"
                            f"梯度：{grad}"
                        )

                # 4. 应用梯度到参数
                self.actor_optimizer.zero_grad()
                for param, grad in zip(self.actor.parameters(), combined_grad):
                    if param.grad is None:
                        param.grad = grad.clone()
                    else:
                        param.grad.copy_(grad)

            # filter_grad_indx = range(self.ppo_epoch)
            # sol, _ = MinNormSolver.find_min_norm_element([policy_grads[t] for t in filter_grad_indx])
            # grads = policy_grads[0]
            # j = 0
            # init = True
            # for i in filter_grad_indx:
            #     for g1, g2 in zip(grads, policy_grads[i]):
            #         if init:
            #             g1 = sol[j] * g2
            #             init = False
            #         else:
            #             g1 += sol[j] * g2
            #     j += 1
            #
            # i = 0
            # for param in self.actor.parameters():
            #     if param.grad is not None:
            #         param.grad = grads[i]
            #         i += 1
            # assert i == len(grads)

            if self.use_max_grad_norm:
                actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            else:
                actor_grad_norm = get_grad_norm(self.actor.parameters())

            train_info["actor_grad_norm"] = actor_grad_norm

            self.actor_optimizer.step()

        #########################################################################################################

        num_updates = self.ppo_epoch * self.actor_num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        if self.use_mgda:
            return policy_loss_epoch, dist_entropy_epoch, imp_weights_epoch, optimizer, train_info
        else:
            return train_info



    def lr_decay(self, episode, episodes):
        """Decay the learning rates.
        Args:
            episode: (int) current training episode.
            episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)

    def get_actions(
            self, obs, rnn_states_actor, masks, available_actions=None, last_actions=None,
            deterministic=False,
            n_agents=None, agent_id=None, all_obs=None,
            all_actions=None,
    ):
        """Compute actions for the given inputs.
        Args:
            obs: (np.ndarray) local agent inputs to the actor.
            rnn_states_actor: (np.ndarray) if actor has RNN layer, RNN states for actor.
            masks: (np.ndarray) denotes points at which RNN states should be reset.
            available_actions: (np.ndarray) denotes which actions are available to agent
                                 (if None, all actions available)
            deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        ### get the sequence

        self.n_agents = n_agents
        self.n_rollout_threads = obs.shape[0]

        all_obs_ = check(np.array(all_obs).transpose(1, 0, 2)).cuda()
        all_actions = check(np.array(all_actions).transpose(1, 0, 2)).cuda()

        actions, action_log_probs, rnn_states_actor_, = self.actor(obs, rnn_states_actor,
                                                                   masks,
                                                                   available_actions,
                                                                   deterministic, all_obs_, all_actions
                                                                   )

        return actions, action_log_probs, rnn_states_actor_,

    def evaluate_actions(
            self,
            obs,
            rnn_states_actor,
            action,
            masks,
            available_actions=None,
            active_masks=None,
            all_obs=None,
            all_actions=None
    ):
        """Get action logprobs, entropy, and distributions for actor update.
        Args:
            obs: (np.ndarray / torch.Tensor) local agent inputs to the actor.
            rnn_states_actor: (np.ndarray / torch.Tensor) if actor has RNN layer, RNN states for actor.
            action: (np.ndarray / torch.Tensor) actions whose log probabilities and entropy to compute.
            masks: (np.ndarray / torch.Tensor) denotes points at which RNN states should be reset.
            available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                    (if None, all actions available)
            active_masks: (np.ndarray / torch.Tensor) denotes whether an agent is active or dead.
        """

        (action_log_probs, dist_entropy, action_distribution,) = self.actor.evaluate_actions(
            obs, rnn_states_actor, action, masks, available_actions, active_masks, all_obs, all_actions

        )
        return action_log_probs, dist_entropy, action_distribution

    def act(
            self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False, all_obs=None,
            all_actions=None
    ):
        """Compute actions using the given inputs.
        Args:
            obs: (np.ndarray) local agent inputs to the actor.
            rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
            masks: (np.ndarray) denotes points at which RNN states should be reset.
            available_actions: (np.ndarray) denotes which actions are available to agent
                                    (if None, all actions available)
            deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """

        actions, action_log_probs, rnn_states_actor_, = self.actor(obs,
                                                                   rnn_states_actor,
                                                                   masks,
                                                                   available_actions,
                                                                   deterministic, all_obs, all_actions)

        return actions, rnn_states_actor_

    def compute_grad_weights(self, policy_grads_list):
        """计算每个目标的梯度权重，动态调整权重以平衡多个目标"""
        # 假设这里每个目标的梯度是一个张量列表，每个目标对应一组梯度
        grad_norms = []

        # 计算每个目标的梯度L2范数（即梯度的大小）
        for grads in policy_grads_list:
            norm = 0.0
            for grad in grads:
                norm += grad.norm(p=2) ** 2  # 计算每个梯度的L2范数并求和
            grad_norms.append(norm.sqrt())  # 对每个目标的梯度计算L2范数

        # 归一化梯度范数：我们将梯度范数归一化到 [0, 1] 范围内，避免数值差异太大
        total_norm = sum(grad_norms)
        if total_norm == 0:
            return [1.0 / len(grad_norms)] * len(grad_norms)  # 如果总范数为0，则均分权重

        # 计算每个目标的相对权重
        weights = [norm / total_norm for norm in grad_norms]

        # 返回加权后的权重列表
        return weights


