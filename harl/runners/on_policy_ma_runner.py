"""Runner for on-policy MA algorithms."""
import numpy as np
import torch
from harl.runners.on_policy_base_runner import OnPolicyBaseRunner

from torch.autograd import Variable
from harl.utils.mgpo.min_norm_solvers import MinNormSolver, gradient_normalizers
import torch.nn as nn
from harl.utils.envs_tools import check
from harl.utils.models_tools import get_grad_norm


class OnPolicyMARunner(OnPolicyBaseRunner):
    """Runner for on-policy MA algorithms."""

    def train(self, episode=0):
        """Training procedure for MAPPO."""
        actor_train_infos = []
        policy_grads = {}
        all_gradients = []
        all_optimizer = []
        all_loss = []
        all_entropy = []
        all_weights = []
        # compute advantages
        if self.value_normalizer is not None:
            advantages = self.critic_buffer.returns[
                         :-1
                         ] - self.value_normalizer.denormalize(self.critic_buffer.value_preds[:-1])
        else:
            advantages = (
                    self.critic_buffer.returns[:-1] - self.critic_buffer.value_preds[:-1]
            )

        # normalize advantages for FP
        if self.state_type == "FP":
            active_masks_collector = [
                self.actor_buffer[i].active_masks for i in range(self.num_agents)
            ]
            active_masks_array = np.stack(active_masks_collector, axis=2)
            advantages_copy = advantages.copy()
            advantages_copy[active_masks_array[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        # update actors
        if self.share_param:
            if self.use_mgda:
                for agent_id in range(self.num_agents):
                    policy_loss, dist_entropy, imp_weights, optimizer, parameters, actor_train_info = self.actor[
                        0].share_param_train(
                        self.actor_buffer, advantages.copy(), self.num_agents, self.state_type)
                    actor_train_infos.append(actor_train_info)
                    all_gradients.append(parameters)
                    all_optimizer.append(optimizer)
                    all_loss.append(policy_loss)
                    all_entropy.append(dist_entropy)
                    all_weights.append(imp_weights)

                    policy_grads[agent_id] = []
                    for param in parameters:  # 遍历 parameters
                        if param.grad is not None:
                            policy_grads[agent_id].append(Variable(param.grad.data.clone(), requires_grad=False))

                filter_grad_indx = range(self.num_agents)
                sol, _ = MinNormSolver.find_min_norm_element([policy_grads[t] for t in filter_grad_indx])

                grads = [g.clone() for g in policy_grads[agent_id]]  # 保持原始的grad拷贝
                j = 0
                init = True
                for i in filter_grad_indx:
                    for g1, g2 in zip(grads, policy_grads[i]):
                        if init:
                            g1 = sol[j] * g2  # g1 += sol[j] * g2
                            init = False
                        else:
                            g1 += sol[j] * g2
                    j += 1
                i = 0
                for parameters in all_gradients:
                    for p, param in enumerate(parameters):
                        if param.grad is not None:
                            param.grad = grads[i]  # grads[i][p]
                            i += 1
                # assert i == len(grads[i])

                all_norm = []
                for agent_id in range(self.num_agents):
                    if self.use_max_grad_norm:
                        actor_grad_norm = nn.utils.clip_grad_norm_(all_gradients[agent_id], self.max_grad_norm)
                    else:
                        actor_grad_norm = get_grad_norm(all_gradients[agent_id])
                    all_optimizer[agent_id].step()
                    all_norm.append(actor_grad_norm)
                    actor_train_infos[agent_id]["actor_grad_norm"] = actor_grad_norm
            else:
                actor_train_info = self.actor[0].share_param_train(
                    self.actor_buffer, advantages.copy(), self.num_agents, self.state_type
                )
            for _ in torch.randperm(self.num_agents):
                actor_train_infos.append(actor_train_info)
        else:
            if self.use_mgda:
                for agent_id in range(self.num_agents):
                    if self.state_type == "EP":
                        policy_loss, dist_entropy, imp_weights, optimizer, parameters, actor_train_info = self.actor[
                            agent_id].train(self.actor_buffer[agent_id], advantages.copy(), "EP")
                    elif self.state_type == "FP":
                        policy_loss, dist_entropy, imp_weights, optimizer, parameters, actor_train_info = self.actor[
                            agent_id].train(self.actor_buffer[agent_id],
                                            advantages[:, :, agent_id].copy(),
                                            "FP", )
                    all_gradients.append(parameters)
                    all_optimizer.append(optimizer)
                    all_loss.append(policy_loss)
                    all_entropy.append(dist_entropy)
                    all_weights.append(imp_weights)
                    actor_train_infos.append(actor_train_info)

                    policy_grads[agent_id] = []
                    for param in parameters:  # 遍历 parameters
                        if param.grad is not None:
                            policy_grads[agent_id].append(Variable(param.grad.data.clone(), requires_grad=False))

                filter_grad_indx = range(self.num_agents)
                sol, _ = MinNormSolver.find_min_norm_element([policy_grads[t] for t in filter_grad_indx])

                grads = [g.clone() for g in policy_grads[0]]  # 保持原始的grad拷贝
                j = 0
                init = True
                for i in filter_grad_indx:
                    for g1, g2 in zip(grads, policy_grads[i]):
                        if init:
                            g1 = sol[j] * g2  # g1 += sol[j] * g2
                            init = False
                        else:
                            g1 += sol[j] * g2
                    j += 1
                i = 0
                for parameters in all_gradients:
                    for p, param in enumerate(parameters):
                        if param.grad is not None:
                            param.grad = grads[i]  # grads[i][p]
                            i += 1
                # assert i == len(grads[i])

                all_norm = []
                for agent_id in range(self.num_agents):
                    if self.use_max_grad_norm:
                        actor_grad_norm = nn.utils.clip_grad_norm_(all_gradients[agent_id], self.max_grad_norm)
                    else:
                        actor_grad_norm = get_grad_norm(all_gradients[agent_id])
                    actor_train_infos[agent_id]["actor_grad_norm"] = actor_grad_norm
                    all_optimizer[agent_id].step()
                    all_norm.append(actor_grad_norm)
            else:
                for agent_id in range(self.num_agents):
                    if self.state_type == "EP":
                        actor_train_info = self.actor[agent_id].train(
                            self.actor_buffer[agent_id], advantages.copy(), "EP"
                        )
                    elif self.state_type == "FP":
                        actor_train_info = self.actor[agent_id].train(
                            self.actor_buffer[agent_id],
                            advantages[:, :, agent_id].copy(),
                            "FP",
                        )
                    actor_train_infos.append(actor_train_info)

        # update critic
        critic_train_info = self.critic.train(self.critic_buffer, self.value_normalizer)

        return actor_train_infos, critic_train_info
