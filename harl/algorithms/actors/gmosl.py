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

        if args["shareGraph"] == False:
            self.graph_actor = Actor_graph(args, self.obs_shape, self.n_actions, self.num_agents, self.device)
            self.graph_actor_optimizer = torch.optim.Adam(self.graph_actor.parameters(),
                                                          lr=self.lr_graph,
                                                          eps=self.opti_eps,
                                                          weight_decay=self.weight_decay)

            self.graph_actor_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.graph_actor_optimizer,
                mode='min',  # 学习率减少基于最小化的监控指标
                factor=0.5,  # 每次减少学习率的比例，默认是 0.8
                patience=20,  # 等待 50 个 epoch 不改善后才减少学习率
                verbose=True,  # 打印学习率调整日志
                threshold=1e-3,  # 只有当损失的变化幅度低于 1e-3 时才触发
                threshold_mode='rel'  # 基于相对变化
            )

        self.G_s = []

        self.actor = StochasticGraphPolicy(args, self.obs_space, self.act_space, self.num_agents, self.device)
        self.actor_optimizer = torch.optim.Adam(
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
        self.chaos_grad = args["chaos_grad"]

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

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
            father_actions_batch,
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
            father_actions=father_actions_batch,
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
        policy_loss.backward(retain_graph=True)

        # actor_model = self.actor
        # if self.args["shareGraph"] == False:
        #     graph_model = self.graph_actor
        # # Calculate parameters for each component (Critic and GRM for this example)
        # actor_params = self.count_parameters(actor_model)
        # if self.args["shareGraph"] == False:
        #     graph_params = self.count_parameters(graph_model)
        #
        # # Print the parameters count
        # print(f"Actor parameters: {actor_params}")
        # if self.args["shareGraph"] == False:
        #     print(f"graph parameters: {graph_params}")

        if self.use_mgda:
            if epoch < self.ppo_epoch and self.chaos_grad:
                if self.use_max_grad_norm:
                    actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                else:
                    actor_grad_norm = get_grad_norm(self.actor.parameters())

                self.actor_optimizer.step()
            return policy_loss, dist_entropy, imp_weights, self.actor_optimizer
        elif self.use_mgda_epoch:
            return policy_loss, dist_entropy, imp_weights, self.actor_optimizer
        else:
            if self.use_max_grad_norm:
                actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            else:
                actor_grad_norm = get_grad_norm(self.actor.parameters())

            self.actor_optimizer.step()

            return policy_loss, dist_entropy, actor_grad_norm, imp_weights

    def train(self, actor_buffer, advantages, state_type, all_obs=None, episode=0, all_actions=None):
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

        # ppo_time = time.time()

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
                    policy_loss, dist_entropy, imp_weights, optimizer = self.update(
                        sample, _)
                    train_info["policy_loss"] += policy_loss.item()
                    train_info["dist_entropy"] += dist_entropy.item()
                    train_info["actor_grad_norm"] += 0
                    train_info["ratio"] += imp_weights.mean()
                    policy_loss_epoch += policy_loss
                    dist_entropy_epoch += dist_entropy
                    imp_weights_epoch += imp_weights
                else:
                    policy_loss, dist_entropy, actor_grad_norm, imp_weights = self.update(
                        sample)

                    train_info["policy_loss"] += policy_loss.item()
                    train_info["dist_entropy"] += dist_entropy.item()
                    train_info["actor_grad_norm"] += actor_grad_norm
                    train_info["ratio"] += imp_weights.mean()

            if self.use_mgda_epoch:
                policy_grads[_] = []
                for param in self.actor.parameters():
                    if param.grad is not None:
                        policy_grads[_].append(Variable(param.grad.data.clone(), requires_grad=False))

        #####################################单独训练图####################################################################
        if self.args["shareGraph"] == False:
            epi_roll = self.args["episode_length"] * self.args["n_rollout_threads"]

            agent_id_graph = torch.eye(self.num_agents, device=self.tpdv['device']).unsqueeze(0).repeat(epi_roll, 1,
                                                                                                        1)  # self.n_rollout_threads, num_agents, num_agents

            all_obs_ = check(all_obs).to(**self.tpdv).reshape(epi_roll, self.num_agents, -1)
            advantages_ = check(advantages).to(**self.tpdv).reshape(epi_roll, -1)

            if self.args["ifLast_action"] == True:
                last_actions_batch_ = check(np.array(actor_buffer[0].last_actions)).to(**self.tpdv).reshape(epi_roll,
                                                                                                            self.num_agents,
                                                                                                            self.n_actions)
            else:
                last_actions_batch_ = check(all_actions).to(**self.tpdv).reshape(epi_roll, self.num_agents, -1)

            inputs_graph = torch.cat((all_obs_, agent_id_graph, last_actions_batch_), -1).float()

            # graph_time = time.time()
            encoder_output, samples, mask_scores, entropy, adj_prob, \
            log_softmax_logits_for_rewards, entropy_regularization, sparse_loss = self.graph_actor(inputs_graph)
            # print(f"graph speeded {time.time() - graph_time}")

            loss_graph_actor = -(torch.sum(log_softmax_logits_for_rewards, dim=(1, 2)) * torch.sum(
                advantages_.unsqueeze(1).repeat(1, self.n_agents, 1), dim=(1, 2))).mean()
            if self.args["sparse_loss"] == True:
                loss_graph_actor += sparse_loss

            ##########################################################
            # epi_roll = self.args["episode_length"] * self.args["n_rollout_threads"]
            # # ## DAG loss
            # loss = 0
            # # mask_scores_tensor = torch.stack(mask_scores).permute(1,0,2)
            # for i in range(epi_roll):
            #     m_s = adj_prob[i]
            #     # sparse_loss = self.args.tau_A * torch.sum(torch.abs(m_s))
            #     h_A = _h_A(m_s, self.num_agents)
            #     loss += h_A
            # loss_hA = loss / epi_roll
            # loss_graph_actor = loss_graph_actor + loss_hA  # 这个loss似乎起穩定器的效果
            #########################################################################################################

            self.graph_actor_optimizer.zero_grad()

            # if self.use_mgda_epoch:
            #     actor_weight, graph_weight = self.pareto_epoch(policy_loss_epoch, loss_graph_actor,
            #                                                    self.actor.parameters(),
            #                                                    self.graph_actor.parameters())

            policy_loss_epoch = policy_loss_epoch + loss_graph_actor
            policy_loss_epoch.backward(retain_graph=True)

            # loss_graph_actor.backward()

            if self.use_max_grad_norm:
                graph_actor_grad_norm = nn.utils.clip_grad_norm_(self.graph_actor.parameters(),
                                                                 self.max_grad_norm)
            else:
                graph_actor_grad_norm = get_grad_norm(self.graph_actor.parameters())

            train_info["loss_graph_actor"] = loss_graph_actor.mean()
            self.graph_actor_optimizer.step()
            if episode > 3000:
                self.graph_actor_lr_scheduler.step(loss_graph_actor)

        #########################################################################################################

        #########################################多目标优化################################################################
        if self.use_mgda_epoch:
            filter_grad_indx = range(self.ppo_epoch)
            # 确保梯度归一化正确
            # grads = [g / (torch.norm(g) + 1e-8) for g in policy_grads]
            # sol, _ = MinNormSolver.find_min_norm_element(grads)
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

    def share_param_train(self, actor_buffer, advantages, num_agents, state_type, all_obs=None, episode=0,
                          all_actions=None, ):
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
        train_info["loss_graph_actor"] = 0

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

        for e in range(self.ppo_epoch):
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

            for _ in range(self.actor_num_mini_batch):
                batches = [[] for _ in range(10)]
                for generator in data_generators:
                    sample = next(generator)
                    for i in range(10):
                        batches[i].append(sample[i])
                for i in range(7):
                    batches[i] = np.concatenate(batches[i], axis=0)
                batches[8] = np.concatenate(batches[8], axis=0)
                if batches[7][0] is None:
                    batches[7] = None
                else:
                    batches[7] = np.concatenate(batches[7], axis=0)

                if self.use_mgda or self.use_mgda_epoch:
                    policy_loss, dist_entropy, imp_weights, optimizer = self.update(
                        tuple(batches), e)
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
                policy_grads[e] = []
                for param in self.actor.parameters():
                    if param.grad is not None:
                        policy_grads[e].append(Variable(param.grad.data.clone(), requires_grad=False))

        #########################################################################################################
        #####################################单独训练图####################################################################
        if self.args["shareGraph"] == False:
            epi_roll = self.args["episode_length"] * self.args["n_rollout_threads"]

            agent_id_graph = torch.eye(self.num_agents, device=self.tpdv['device']).unsqueeze(0).repeat(epi_roll, 1,
                                                                                                        1)  # self.n_rollout_threads, num_agents, num_agents

            all_obs_ = check(all_obs).to(**self.tpdv).reshape(epi_roll, self.num_agents, -1)
            advantages_ = check(advantages).to(**self.tpdv).reshape(epi_roll, -1)
            if self.args["ifLast_action"] == True:
                last_actions_batch_ = check(np.array(actor_buffer[0].last_actions)).to(**self.tpdv).reshape(epi_roll,
                                                                                                            self.num_agents,
                                                                                                            self.n_actions)
            else:
                last_actions_batch_ = check(all_actions).to(**self.tpdv).reshape(epi_roll, self.num_agents, -1)

            inputs_graph = torch.cat((all_obs_, agent_id_graph, last_actions_batch_), -1).float()

            # graph_time = time.time()
            encoder_output, samples, mask_scores, entropy, adj_prob, \
            log_softmax_logits_for_rewards, entropy_regularization, sparse_loss = self.graph_actor(inputs_graph)
            # print(f"graph speeded {time.time() - graph_time}")

            ##########################################################
            # ## DAG loss
            # loss = 0
            # # mask_scores_tensor = torch.stack(mask_scores).permute(1,0,2)
            # for i in range(epi_roll):
            #     m_s = adj_prob[i]
            #     # sparse_loss = self.args.tau_A * torch.sum(torch.abs(m_s))
            #     h_A = _h_A(m_s, self.num_agents)
            #     loss += h_A
            # loss_hA = loss / epi_roll
            # loss_graph_actor = loss_graph_actor + loss_hA  # 这个loss似乎起穩定器的效果

            ##########################################################

            loss_graph_actor = -(torch.sum(log_softmax_logits_for_rewards, dim=(1, 2)) * torch.sum(
                advantages_.unsqueeze(1).repeat(1, self.n_agents, 1), dim=(1, 2))).mean()
            if self.args["sparse_loss"] == True:
                loss_graph_actor += sparse_loss
            self.graph_actor_optimizer.zero_grad()

            policy_loss_epoch += loss_graph_actor
            policy_loss_epoch.backward(retain_graph=True)

            if self.use_max_grad_norm:
                graph_actor_grad_norm = nn.utils.clip_grad_norm_(self.graph_actor.parameters(), self.max_grad_norm)
            else:
                graph_actor_grad_norm = get_grad_norm(self.graph_actor.parameters())

            train_info["loss_graph_actor"] = loss_graph_actor.mean()
            self.graph_actor_optimizer.step()
            # if episode > 5000:
            #     self.graph_actor_lr_scheduler.step(loss_graph_actor)

        #########################################################################################################
        #########################################多目标优化################################################################
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

            train_info["actor_grad_norm"] = actor_grad_norm

        #########################################################################################################
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
        if self.args["shareGraph"] == False:
            update_linear_schedule(self.graph_actor_optimizer, episode, episodes, self.lr_graph)

    def get_actions(
            self, obs, rnn_states_actor, masks, available_actions=None, last_actions=None,
            deterministic=False,
            n_agents=None, agent_id=None, all_obs=None, step_reuse=1,
            graph_actor=None,
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

        agent_id_graph = torch.eye(self.num_agents, device=self.tpdv['device']).unsqueeze(0).repeat(
            self.n_rollout_threads, 1,
            1)  # self.n_rollout_threads, num_agents, num_agents

        all_obs_ = check(np.array(all_obs).transpose(1, 0, 2)).cuda()
        if self.args["ifLast_action"] == True:
            last_actions_ = check(last_actions).cuda()
        else:
            all_actions = check(np.array(all_actions).transpose(1, 0, 2)).cuda()
            last_actions_ = check(all_actions).cuda()

        if self.args["shareGraph"] == False:
            inputs_graph = torch.cat((all_obs_, agent_id_graph, last_actions_), -1).float()  # 1. 4.33
            encoder_output, samples, mask_scores, entropy, adj_prob, \
            log_softmax_logits_for_rewards, entropy_regularization, _ = self.graph_actor(inputs_graph)
            graph_A = samples.clone().cpu().numpy()

        else:
            inputs_graph = torch.cat((all_obs_, agent_id_graph, last_actions_), -1).float()  # 1. 4.33
            encoder_output, samples, mask_scores, entropy, adj_prob, \
            log_softmax_logits_for_rewards, entropy_regularization, _ = graph_actor(inputs_graph)
            graph_A = samples.clone().cpu().numpy()

        ######## pruning
        self.G_s = []
        for i in range(graph_A.shape[0]):
            G = ig.Graph.Weighted_Adjacency(graph_A[i].tolist())
            if not is_acyclic(graph_A[i]):
                G, new_A = pruning_1(G, graph_A[i])
            self.G_s.append(G)

        actions, action_log_probs, rnn_states_actor_, father_actions_ = self.actor(obs,
                                                                                   rnn_states_actor,
                                                                                   masks, self.G_s,
                                                                                   available_actions,
                                                                                   deterministic,
                                                                                   step_reuse=step_reuse
                                                                                   )

        return actions, action_log_probs, rnn_states_actor_, father_actions_

    def evaluate_actions(
            self,
            obs,
            rnn_states_actor,
            action,
            masks,
            available_actions=None,
            active_masks=None,
            father_actions=None,
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
            obs, rnn_states_actor, action, masks, available_actions, active_masks,
            father_actions
        )
        return action_log_probs, dist_entropy, action_distribution

    def act(
            self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False,
            agent_id=None
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

        actions, action_log_probs, rnn_states_actor_, father_actions_ = self.actor(obs,
                                                                                   rnn_states_actor,
                                                                                   masks, self.G_s,
                                                                                   available_actions,
                                                                                   deterministic)

        return actions, rnn_states_actor_



    def pareto_epoch(self, actor_loss, graph_loss, actor_params, graph_params):

        # 梯度归一化
        actor_grads = [p.grad for p in actor_params if p.grad is not None]
        graph_grads = [p.grad for p in graph_params if p.grad is not None]
        actor_grads = [self.normalize_gradients(g) for g in actor_grads]
        graph_grads = [self.normalize_gradients(g) for g in graph_grads]

        # 动态调整任务权重
        actor_weight = actor_loss.item() / (actor_loss.item() + graph_loss.item())
        graph_weight = 1 - actor_weight

        # 计算 Pareto 解
        sol, _ = MinNormSolver.find_min_norm_element([actor_grads, graph_grads])

        # 更新参数
        for i, param in enumerate(actor_params):
            if param.grad is not None:
                param.grad = sol[0] * actor_weight * actor_grads[i]
        for i, param in enumerate(graph_params):
            if param.grad is not None:
                param.grad = sol[1] * graph_weight * graph_grads[i]

        return actor_weight, graph_weight

    def normalize_gradients(self,grads):
        """
        Normalize gradients to avoid exploding/vanishing gradients.
        Args:
            grads: List of gradients.
        Returns:
            Normalized gradients.
        """
        norm = torch.norm(torch.stack([torch.norm(g, p=2) for g in grads]), p=2)
        return [g / (norm + 1e-8) for g in grads]  # 防止除零
