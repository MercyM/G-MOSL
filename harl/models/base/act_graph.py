import torch.nn as nn
from harl.models.base.distributions import Categorical, DiagGaussian
import igraph as ig
import torch
import numpy as np
import torch.nn.functional as f


class ACTGraphLayer(nn.Module):
    """MLP Module to compute actions."""

    def __init__(
            self, n_agents, action_space, inputs_dim, initialization_method, gain, args=None
    ):
        """Initialize ACTLayer.
        Args:
            action_space: (gym.Space) action space.
            inputs_dim: (int) dimension of network input.
            initialization_method: (str) initialization method.
            gain: (float) gain of the output layer of the network.
            args: (dict) arguments relevant to the network.
        """
        super(ACTGraphLayer, self).__init__()
        self.action_space = action_space.__class__.__name__
        self.multidiscrete_action = False

        if action_space.__class__.__name__ == "Discrete":
            action_dim = action_space.n
            self.action_out_old = Categorical(
                inputs_dim, action_dim, initialization_method, gain
            )
            self.action_out = Categorical(
                inputs_dim + action_dim * n_agents * n_agents, action_dim, initialization_method, gain
            )

        elif action_space.__class__.__name__ == "Box":
            action_dim = action_space.shape[0]
            self.action_out = DiagGaussian(
                inputs_dim + action_dim * n_agents * n_agents, action_dim, initialization_method, gain, args
            )
        elif action_space.__class__.__name__ == "MultiDiscrete":
            self.multidiscrete_action = True
            action_dims = action_space.nvec
            action_outs = []
            for action_dim in action_dims:
                action_outs.append(
                    Categorical(inputs_dim + action_dim * n_agents * n_agents, action_dim, initialization_method, gain,
                                args)
                )
            self.action_outs = nn.ModuleList(action_outs)

        self.action_dim = action_dim

        self.n_agents = n_agents

        self.args = args

        # 初始化全局父动作权重
        self.father_action_weights = nn.Parameter(torch.ones(n_agents))

    def forward(self, x, G_s, available_actions=None, deterministic=False, step_reuse=1):
        """Compute actions and action logprobs from given input.
        Args:
            x: (torch.Tensor) input to network.
            available_actions: (torch.Tensor) denotes which actions are available to agent
                                  (if None, all actions available)
            deterministic: (bool) whether to sample from action distribution or return the mode.
        Returns:
            actions: (torch.Tensor) actions to take.
            action_log_probs: (torch.Tensor) log probabilities of taken actions.
        """

        # 判断是否需要重新计算关系图和 father action

        # 重新计算图和 father action
        # if step_reuse % 1 == 0 or step_reuse < 1:
        self.actions_outer, self.action_log_probs_outer = [], [],  # 清空之前存储的father_action_lst
        # 初始化父动作张量
        self.father_action_lst_outer = torch.zeros(self.args["n_rollout_threads"], self.n_agents, self.n_agents * self.action_dim, device=x.device)

        for i, G in enumerate(G_s):
            # 获取拓扑排序
            ordered_vertices = G.topological_sorting()
            n_agents = len(ordered_vertices)
            actions = [0] * self.n_agents
            # 初始化当前图的父动作张量
            father_action_tensor = torch.zeros(n_agents, n_agents * self.action_dim, device=x.device)

            for j in ordered_vertices:
                # 获取当前节点的父节点
                parents = G.neighbors(j, mode=ig.IN)
                if parents:
                    # 获取父节点的动作
                    parent_actions = torch.stack([torch.eye(self.action_dim, device=x.device)[actions[k]] for k in parents])
                    # 如果有权重，应用权重
                    if self.args["father_weight"]:
                        weights = self.father_action_weights[parents]
                        parent_actions = weights.unsqueeze(-1) * parent_actions
                    # 将父节点动作拼接到当前节点的父动作张量中
                    for parent in parents:
                        start_idx = parent * self.action_dim
                        end_idx = (parent + 1) * self.action_dim
                        father_action_tensor[j, start_idx:end_idx] = parent_actions[parents.index(parent)].view(-1)

            # 将当前图的父动作张量添加到批次中
            self.father_action_lst_outer[i] = father_action_tensor


        # for i in range(len(G_s)):
        #     G = G_s[i]
        #     ordered_vertices = G.topological_sorting()
        #     self.n_agents = len(ordered_vertices)
        #     actions = [0] * self.n_agents
        #     father_action_lst = [0] * self.n_agents
        #     for j in ordered_vertices:
        #         father_action_0 = torch.zeros(self.n_agents, self.action_dim)
        #         parents = G.neighbors(j, mode=ig.IN)
        #         if len(parents) != 0:
        #             for k in parents:
        #                 father_act = torch.eye(self.action_dim)[actions[k]]
        #                 if self.args["father_weight"]==True:
        #                     weight = self.father_action_weights[k]  # 使用权重
        #                     father_action_0[k] = weight.cpu() * father_act.detach()
        #                 else:
        #                     father_action_0[k] = father_act.detach()
        #
        #         father_action = father_action_0.reshape(-1)
        #         father_action_lst[j] = father_action
        #
        #     father_action_tensor = torch.stack([item.detach() for item in father_action_lst]).cuda()
        #     self.father_action_lst_outer.append(father_action_tensor)
        #
        # # print(father_action_tensor)
        # self.father_action_lst_outer = torch.stack([item.detach() for item in self.father_action_lst_outer]).cuda()
        self.father_action_shape = self.father_action_lst_outer.shape[-1] * self.father_action_lst_outer.shape[-2]

        x_ = torch.cat((x.squeeze(),
                        self.father_action_lst_outer.reshape(self.args["n_rollout_threads"], -1)),
                       dim=-1)

        if self.multidiscrete_action:
            actions = []
            action_log_probs = []
            for action_out in self.action_outs:
                action_distribution = action_out(x_, available_actions)
                action = (
                    action_distribution.mode()
                    if deterministic
                    else action_distribution.sample()
                )
                action_log_prob = action_distribution.log_probs(action)
                actions.append(action)
                action_log_probs.append(action_log_prob)
            actions = torch.cat(actions, dim=-1)
            action_log_probs = torch.cat(action_log_probs, dim=-1).sum(
                dim=-1, keepdim=True
            )
        elif self.action_space == "Box":
            action_distribution = self.action_out(x_, available_actions)

            actions = (
                action_distribution.mode()
                if deterministic
                else action_distribution.sample()
            )
            action_log_probs = action_distribution.log_probs(actions)

        else:
            action_distribution = self.action_out(x_, available_actions.squeeze())

            actions = (
                action_distribution.mode()
                if deterministic
                else action_distribution.sample()
            )
            action_log_probs = action_distribution.log_probs(actions)

        return actions, action_log_probs, self.father_action_lst_outer.view(-1, self.father_action_shape)

    def evaluate_actions(self, x, action, available_actions=None, active_masks=None, father_action=None):
        """Compute action log probability, distribution entropy, and action distribution.
        Args:
            x: (torch.Tensor) input to network.
            action: (torch.Tensor) actions whose entropy and log probability to evaluate.
            available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
            active_masks: (torch.Tensor) denotes whether an agent is active or dead.
        Returns:
            action_log_probs: (torch.Tensor) log probabilities of the input actions.
            dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
            action_distribution: (torch.distributions) action distribution.
        """

        x_ = torch.cat((x, father_action), dim=-1)
        if self.multidiscrete_action:
            action = torch.transpose(action, 0, 1)
            action_log_probs = []
            dist_entropy = []
            for action_out, act in zip(self.action_outs, action):
                action_distribution = action_out(x_)
                action_log_probs.append(
                    action_distribution.log_probs(act.unsqueeze(-1))
                )
                if active_masks is not None:
                    dist_entropy.append(
                        (action_distribution.entropy() * active_masks)
                        / active_masks.sum()
                    )
                else:
                    dist_entropy.append(
                        action_distribution.entropy() / action_log_probs[-1].size(0)
                    )
            action_log_probs = torch.cat(action_log_probs, dim=-1).sum(
                dim=-1, keepdim=True
            )
            dist_entropy = (
                torch.cat(dist_entropy, dim=-1).sum(dim=-1, keepdim=True).mean()
            )
            return action_log_probs, dist_entropy, None
        else:
            action_distribution = self.action_out(x_, available_actions)

            action_log_probs = action_distribution.log_probs(action)
            if active_masks is not None:
                if self.action_space == "Discrete":
                    dist_entropy = (
                                           action_distribution.entropy() * active_masks.squeeze(-1)
                                   ).sum() / active_masks.sum()
                else:
                    dist_entropy = (
                                           action_distribution.entropy() * active_masks
                                   ).sum() / active_masks.sum()
            else:
                dist_entropy = action_distribution.entropy().mean()

            return action_log_probs, dist_entropy, action_distribution
