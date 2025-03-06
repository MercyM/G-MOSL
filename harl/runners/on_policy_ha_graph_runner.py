"""Runner for on-policy HARL algorithms."""
import numpy as np
import torch
from harl.utils.trans_tools import _t2n
from harl.runners.on_policy_base_runner import OnPolicyBaseRunner

from torch.autograd import Variable
from harl.utils.mgpo.min_norm_solvers import MinNormSolver, gradient_normalizers
import torch.nn as nn
from harl.utils.models_tools import get_grad_norm


class OnPolicyHAGraphRunner(OnPolicyBaseRunner):
    """Runner for on-policy HA algorithms."""

    def train(self, episode=0):
        """Train the model."""
        actor_train_infos = []

        policy_grads = {}
        all_gradients = []
        all_optimizer = []
        all_loss = []
        all_entropy = []
        all_weights = []

        all_obs = []
        for agent_id in range(self.num_agents):
            all_obs.append(self.actor_buffer[agent_id].obs)
        all_obs = np.array(all_obs).transpose(1, 2, 0, 3)[1:, :, :, :]

        all_actions = []
        for agent_id in range(self.num_agents):
            all_actions.append(self.actor_buffer[agent_id].actions)
        all_actions = np.array(all_actions).transpose(1, 2, 0, 3)

        # factor is used for considering updates made by previous agents
        factor = np.ones(
            (
                self.algo_args["train"]["episode_length"],
                self.algo_args["train"]["n_rollout_threads"],
                1,
            ),
            dtype=np.float32,
        )

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

        if self.fixed_order:
            agent_order = list(range(self.num_agents))
        else:
            agent_order = list(torch.randperm(self.num_agents).numpy())
        for agent_id in agent_order:
            self.actor_buffer[agent_id].update_factor(
                factor
            )  # current actor save factor

            # the following reshaping combines the first two dimensions (i.e. episode_length and n_rollout_threads) to form a batch
            available_actions = (
                None
                if self.actor_buffer[agent_id].available_actions is None
                else self.actor_buffer[agent_id]
                         .available_actions[:-1]
                         .reshape(-1, *self.actor_buffer[agent_id].available_actions.shape[2:])
            )

            # compute action log probs for the actor before update.
            old_actions_logprob, _, _ = self.actor[agent_id].evaluate_actions(
                self.actor_buffer[agent_id]
                    .obs[:-1]
                    .reshape(-1, *self.actor_buffer[agent_id].obs.shape[2:]),
                self.actor_buffer[agent_id]
                    .rnn_states[0:1]
                    .reshape(-1, *self.actor_buffer[agent_id].rnn_states.shape[2:]),
                self.actor_buffer[agent_id].actions.reshape(
                    -1, *self.actor_buffer[agent_id].actions.shape[2:]
                ),
                self.actor_buffer[agent_id]
                    .masks[:-1]
                    .reshape(-1, *self.actor_buffer[agent_id].masks.shape[2:]),
                available_actions,
                self.actor_buffer[agent_id]
                    .active_masks[:-1]
                    .reshape(-1, *self.actor_buffer[agent_id].active_masks.shape[2:]),
                self.actor_buffer[agent_id]
                    .father_actions
                    .reshape(-1, *self.actor_buffer[agent_id].father_actions.shape[2:]),
            )

            if self.use_mgda:
                for agent_id in range(self.num_agents):
                    if self.state_type == "EP":
                        policy_loss, dist_entropy, imp_weights, optimizer, parameters, actor_train_info = \
                            self.actor[
                                agent_id].train(self.actor_buffer[agent_id], advantages.copy(), "EP", all_obs=all_obs,
                                                episode=episode,
                                                all_actions=all_actions)
                    elif self.state_type == "FP":
                        policy_loss, dist_entropy, imp_weights, optimizer, parameters, actor_train_info = \
                            self.actor[
                                agent_id].train(self.actor_buffer[agent_id],
                                                advantages[:, :, agent_id].copy(),
                                                "FP", all_obs=all_obs, episode=episode, all_actions=all_actions)

                    all_gradients.append(parameters)
                    all_optimizer.append(optimizer)
                    all_loss.append(policy_loss)
                    all_entropy.append(dist_entropy)
                    all_weights.append(imp_weights)
                    actor_train_infos.append(actor_train_info)
                    # all_graph_loss.append(loss_graph_actor)
                    # all_graph_optimizer.append(graph_actor_optimizer)
                    # all_graph_actor.append(graph_actor)

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
                    # (all_loss[agent_id] + all_graph_loss[agent_id]).backward(retain_graph=True)
                    if self.use_max_grad_norm:
                        actor_grad_norm = nn.utils.clip_grad_norm_(all_gradients[agent_id], self.max_grad_norm)
                        # graph_actor_grad_norm = nn.utils.clip_grad_norm_(all_graph_actor[agent_id].parameters(),
                        #                                                  self.max_grad_norm)
                    else:
                        actor_grad_norm = get_grad_norm(all_gradients[agent_id])
                        # graph_actor_grad_norm = get_grad_norm(all_graph_actor[agent_id].parameters())
                    # all_graph_optimizer[agent_id].step()
                    all_optimizer[agent_id].step()
                    all_norm.append(actor_grad_norm)
            else:
                # update actor
                if self.state_type == "EP":
                    actor_train_info = self.actor[agent_id].train(
                        self.actor_buffer[agent_id], advantages.copy(), "EP"
                    )
                elif self.state_type == "FP":
                    actor_train_info = self.actor[agent_id].train(
                        self.actor_buffer[agent_id], advantages[:, :, agent_id].copy(), "FP"
                    )

            # compute action log probs for updated agent
            new_actions_logprob, _, _ = self.actor[agent_id].evaluate_actions(
                self.actor_buffer[agent_id]
                    .obs[:-1]
                    .reshape(-1, *self.actor_buffer[agent_id].obs.shape[2:]),
                self.actor_buffer[agent_id]
                    .rnn_states[0:1]
                    .reshape(-1, *self.actor_buffer[agent_id].rnn_states.shape[2:]),
                self.actor_buffer[agent_id].actions.reshape(
                    -1, *self.actor_buffer[agent_id].actions.shape[2:]
                ),
                self.actor_buffer[agent_id]
                    .masks[:-1]
                    .reshape(-1, *self.actor_buffer[agent_id].masks.shape[2:]),
                available_actions,
                self.actor_buffer[agent_id]
                    .active_masks[:-1]
                    .reshape(-1, *self.actor_buffer[agent_id].active_masks.shape[2:]),
                self.actor_buffer[agent_id]
                    .father_actions
                    .reshape(-1, *self.actor_buffer[agent_id].father_actions.shape[2:]),
            )

            # update factor for next agent
            factor = factor * _t2n(
                getattr(torch, self.action_aggregation)(
                    torch.exp(new_actions_logprob - old_actions_logprob), dim=-1
                ).reshape(
                    self.algo_args["train"]["episode_length"],
                    self.algo_args["train"]["n_rollout_threads"],
                    1,
                )
            )
            actor_train_infos.append(actor_train_info)

        # update critic
        critic_train_info = self.critic.train(self.critic_buffer, self.value_normalizer)

        if self.algo_args["algo"]["shareGraph"] == True:
            self.TrainIndGraph(all_obs, advantages.copy(), all_actions, critic_train_info, episode)

        return actor_train_infos, critic_train_info
