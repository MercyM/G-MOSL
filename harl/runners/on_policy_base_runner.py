"""Base runner for on-policy algorithms."""

import time
import numpy as np
import torch
import setproctitle
from harl.common.valuenorm import ValueNorm
from harl.common.buffers.on_policy_actor_buffer import OnPolicyActorBuffer
from harl.common.buffers.on_policy_actor_graph_buffer import OnPolicyActorGraphBuffer
from harl.common.buffers.on_policy_critic_buffer_ep import OnPolicyCriticBufferEP
from harl.common.buffers.on_policy_critic_buffer_fp import OnPolicyCriticBufferFP
from harl.algorithms.actors import ALGO_REGISTRY
from harl.algorithms.critics.v_critic import VCritic
from harl.utils.trans_tools import _t2n
from harl.utils.envs_tools import (
    make_eval_env,
    make_train_env,
    make_render_env,
    set_seed,
    get_num_agents,
)
from harl.utils.models_tools import init_device
from harl.utils.configs_tools import init_dir, save_config
from harl.envs import LOGGER_REGISTRY

from harl.algorithms.comm.graph_net_trans import Actor_graph
from harl.utils.models_tools import update_linear_schedule
from harl.algorithms.comm.util import *
from harl.models.policy_models.stochastic_graph_policy import StochasticGraphPolicy
from harl.utils.envs_tools import get_shape_from_obs_space, get_shape_from_act_space
from harl.utils.models_tools import get_grad_norm


class OnPolicyBaseRunner:
    """Base runner for on-policy algorithms."""

    def __init__(self, args, algo_args, env_args):
        """Initialize the OnPolicyBaseRunner class.
        Args:
            args: command-line arguments parsed by argparse. Three keys: algo, env, exp_name.
            algo_args: arguments related to algo, loaded from config file and updated with unparsed command-line arguments.
            env_args: arguments related to env, loaded from config file and updated with unparsed command-line arguments.
        """
        self.args = args
        self.algo_args = algo_args
        self.env_args = env_args

        self.hidden_sizes = algo_args["model"]["hidden_sizes"]
        self.rnn_hidden_size = self.hidden_sizes[-1]
        self.recurrent_n = algo_args["model"]["recurrent_n"]
        self.action_aggregation = algo_args["algo"]["action_aggregation"]
        self.state_type = env_args.get("state_type", "EP")
        self.share_param = algo_args["algo"]["share_param"]
        self.fixed_order = algo_args["algo"]["fixed_order"]
        self.use_mgda = algo_args["algo"]["use_mgda"]
        self.use_max_grad_norm = algo_args["algo"]["use_max_grad_norm"]
        self.max_grad_norm = algo_args["algo"]["max_grad_norm"]
        self.ppo_epoch = algo_args["algo"]["ppo_epoch"]
        self.actor_num_mini_batch = algo_args["algo"]["actor_num_mini_batch"]
        set_seed(algo_args["seed"])
        self.device = init_device(algo_args["device"])
        if not self.algo_args["render"]["use_render"]:  # train, not render
            self.run_dir, self.log_dir, self.save_dir, self.writter = init_dir(
                args["env"],
                env_args,
                args["algo"],
                args["exp_name"],
                algo_args["seed"]["seed"],
                logger_path=algo_args["logger"]["log_dir"],
            )
            save_config(args, algo_args, env_args, self.run_dir)
        # set the title of the process
        setproctitle.setproctitle(
            str(args["algo"]) + "-" + str(args["env"]) + "-" + str(args["exp_name"])
        )

        # set the config of env
        if self.algo_args["render"]["use_render"]:  # make envs for rendering
            (
                self.envs,
                self.manual_render,
                self.manual_expand_dims,
                self.manual_delay,
                self.env_num,
            ) = make_render_env(args["env"], algo_args["seed"]["seed"], env_args)
        else:  # make envs for training and evaluation
            self.envs = make_train_env(
                args["env"],
                algo_args["seed"]["seed"],
                algo_args["train"]["n_rollout_threads"],
                env_args,
            )
            self.eval_envs = (
                make_eval_env(
                    args["env"],
                    algo_args["seed"]["seed"],
                    algo_args["eval"]["n_eval_rollout_threads"],
                    env_args,
                )
                if algo_args["eval"]["use_eval"]
                else None
            )
        self.num_agents = get_num_agents(args["env"], env_args, self.envs)

        print("share_observation_space: ", self.envs.share_observation_space)
        print("observation_space: ", self.envs.observation_space)
        print("action_space: ", self.envs.action_space)

        # actor
        if self.args["algo"] == "gmosl" or self.args["algo"] == "happo_graph" or self.args[
            "algo"] == "haa2c_graph" or self.args["algo"] == "hatrpo_graph":
            # actor
            if self.share_param:
                self.actor = []
                agent = ALGO_REGISTRY[args["algo"]](
                    {**algo_args["model"], **algo_args["algo"], **algo_args["train"]},
                    self.envs.observation_space[0],
                    self.envs.action_space[0],
                    self.num_agents,
                    0,
                    device=self.device,
                )
                self.actor.append(agent)
                for agent_id in range(1, self.num_agents):
                    assert (
                            self.envs.observation_space[agent_id]
                            == self.envs.observation_space[0]
                    ), "Agents have heterogeneous observation spaces, parameter sharing is not valid."
                    assert (
                            self.envs.action_space[agent_id] == self.envs.action_space[0]
                    ), "Agents have heterogeneous action spaces, parameter sharing is not valid."
                    self.actor.append(self.actor[0])
            else:
                self.actor = []
                for agent_id in range(self.num_agents):
                    agent = ALGO_REGISTRY[args["algo"]](
                        {**algo_args["model"], **algo_args["algo"], **algo_args["train"]},
                        self.envs.observation_space[agent_id],
                        self.envs.action_space[agent_id],
                        self.num_agents,
                        agent_id,
                        device=self.device,
                    )
                    self.actor.append(agent)
        else:
            if self.share_param:
                self.actor = []
                agent = ALGO_REGISTRY[args["algo"]](
                    {**algo_args["model"], **algo_args["algo"]},
                    self.envs.observation_space[0],
                    self.envs.action_space[0],
                    device=self.device,
                )
                self.actor.append(agent)
                for agent_id in range(1, self.num_agents):
                    assert (
                            self.envs.observation_space[agent_id]
                            == self.envs.observation_space[0]
                    ), "Agents have heterogeneous observation spaces, parameter sharing is not valid."
                    assert (
                            self.envs.action_space[agent_id] == self.envs.action_space[0]
                    ), "Agents have heterogeneous action spaces, parameter sharing is not valid."
                    self.actor.append(self.actor[0])
            else:
                self.actor = []
                for agent_id in range(self.num_agents):
                    agent = ALGO_REGISTRY[args["algo"]](
                        {**algo_args["model"], **algo_args["algo"]},
                        self.envs.observation_space[agent_id],
                        self.envs.action_space[agent_id],
                        device=self.device,
                    )
                    self.actor.append(agent)

        if self.algo_args["render"]["use_render"] is False:  # train, not render
            self.actor_buffer = []
            if args["algo"] == "gmosl" or self.args["algo"] == "happo_graph" or self.args[
                "algo"] == "haa2c_graph" or self.args["algo"] == "hatrpo_graph":
                for agent_id in range(self.num_agents):
                    ac_bu = OnPolicyActorGraphBuffer(
                        {**algo_args["train"], **algo_args["model"]},
                        self.envs.observation_space[agent_id],
                        self.envs.action_space[agent_id],
                        self.num_agents
                    )
                    self.actor_buffer.append(ac_bu)
            else:
                for agent_id in range(self.num_agents):
                    ac_bu = OnPolicyActorBuffer(
                        {**algo_args["train"], **algo_args["model"]},
                        self.envs.observation_space[agent_id],
                        self.envs.action_space[agent_id],
                    )
                    self.actor_buffer.append(ac_bu)

            share_observation_space = self.envs.share_observation_space[0]
            self.critic = VCritic(
                {**algo_args["model"], **algo_args["algo"]},
                share_observation_space,
                device=self.device,
            )
            if self.state_type == "EP":
                # EP stands for Environment Provided, as phrased by MAPPO paper.
                # In EP, the global states for all agents are the same.
                self.critic_buffer = OnPolicyCriticBufferEP(
                    {**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
                    share_observation_space,
                )
            elif self.state_type == "FP":
                # FP stands for Feature Pruned, as phrased by MAPPO paper.
                # In FP, the global states for all agents are different, and thus needs the dimension of the number of agents.
                self.critic_buffer = OnPolicyCriticBufferFP(
                    {**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
                    share_observation_space,
                    self.num_agents,
                )
            else:
                raise NotImplementedError

            if self.algo_args["train"]["use_valuenorm"] is True:
                self.value_normalizer = ValueNorm(1, device=self.device)
            else:
                self.value_normalizer = None

            self.logger = LOGGER_REGISTRY[args["env"]](
                args, algo_args, env_args, self.num_agents, self.writter, self.run_dir
            )
        if self.algo_args["train"]["model_dir"] is not None:  # restore model
            self.restore()

        if args["algo"] == "gmosl" or self.args["algo"] == "happo_graph" or self.args[
            "algo"] == "haa2c_graph" or self.args["algo"] == "hatrpo_graph":
            if algo_args["algo"]["shareGraph"] == True:
                self.lr_graph = algo_args["model"]["lr_graph"]
                self.opti_eps = algo_args["model"]["opti_eps"]
                self.weight_decay = algo_args["model"]["weight_decay"]

                self.n_actions = get_shape_from_act_space(self.envs.action_space[0])
                self.obs_shape = get_shape_from_act_space(self.envs.observation_space[0])

                self.graph_actor = Actor_graph(algo_args["algo"], self.obs_shape, self.n_actions, self.num_agents,
                                               self.device)
                self.graph_actor_optimizer = torch.optim.Adam(self.graph_actor.parameters(),
                                                              lr=self.lr_graph,
                                                              eps=self.opti_eps,
                                                              weight_decay=self.weight_decay)

                self.tpdv = dict(dtype=torch.float32, device=self.device)
                self.graph_actor_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.graph_actor_optimizer,
                    mode='min',  # 学习率减少基于最小化的监控指标
                    factor=0.8,  # 每次减少学习率的比例，默认是 0.5
                    patience=50,  # 等待 20 个 epoch 不改善后才减少学习率
                    verbose=True,  # 打印学习率调整日志
                    threshold=1e-3,  # 只有当损失的变化幅度低于 1e-3 时才触发
                    threshold_mode='rel'  # 基于相对变化
                )

    def run(self):
        """Run the training (or rendering) pipeline."""
        if self.algo_args["render"]["use_render"] is True:
            self.render()
            return
        print("start running")
        self.warmup()

        episodes = (
                int(self.algo_args["train"]["num_env_steps"])
                // self.algo_args["train"]["episode_length"]
                // self.algo_args["train"]["n_rollout_threads"]
        )

        self.logger.init(episodes)  # logger callback at the beginning of training

        # start_time = time.time()
        for episode in range(1, episodes + 1):
            if self.algo_args["train"][
                "use_linear_lr_decay"
            ]:  # linear decay of learning rate
                if self.share_param:
                    self.actor[0].lr_decay(episode, episodes)
                else:
                    for agent_id in range(self.num_agents):
                        self.actor[agent_id].lr_decay(episode, episodes)
                self.critic.lr_decay(episode, episodes)

            self.logger.episode_init(
                episode
            )  # logger callback at the beginning of each episode

            # if episode > 1:
            #     print(f"episode{episode - 1} spended {time.time() - start_time}")
            #     start_time = time.time()

            self.prep_rollout()  # change to eval mode
            # sample_time = time.time()
            for step in range(self.algo_args["train"]["episode_length"]):
                # Sample actions from actors and values from critics
                if self.args["algo"] == "gmosl" or self.args["algo"] == "happo_graph" or self.args[
                    "algo"] == "haa2c_graph" or self.args["algo"] == "hatrpo_graph":
                    (
                        values,
                        actions,
                        action_log_probs,
                        rnn_states,
                        rnn_states_critic,
                        father_actions,
                        last_actions
                    ) = self.collectGraph(step)
                else:
                    (
                        values,
                        actions,
                        action_log_probs,
                        rnn_states,
                        rnn_states_critic,
                    ) = self.collect(step)
                # actions: (n_threads, n_agents, action_dim)
                (
                    obs,
                    share_obs,
                    rewards,
                    dones,
                    infos,
                    available_actions,
                ) = self.envs.step(actions)
                # obs: (n_threads, n_agents, obs_dim)
                # share_obs: (n_threads, n_agents, share_obs_dim)
                # rewards: (n_threads, n_agents, 1)
                # dones: (n_threads, n_agents)
                # infos: (n_threads)
                # available_actions: (n_threads, ) of None or (n_threads, n_agents, action_number)
                if self.args["algo"] == "gmosl" or self.args["algo"] == "happo_graph" or self.args[
                    "algo"] == "haa2c_graph" or self.args["algo"] == "hatrpo_graph":
                    data = (
                        obs,
                        share_obs,
                        rewards,
                        dones,
                        infos,
                        available_actions,
                        values,
                        actions,
                        action_log_probs,
                        rnn_states,
                        rnn_states_critic,
                        father_actions,
                        last_actions,
                    )
                else:
                    data = (
                        obs,
                        share_obs,
                        rewards,
                        dones,
                        infos,
                        available_actions,
                        values,
                        actions,
                        action_log_probs,
                        rnn_states,
                        rnn_states_critic,
                    )

                self.logger.per_step(data)  # logger callback at each step

                if self.args["algo"] == "gmosl" or self.args["algo"] == "happo_graph" or self.args[
                    "algo"] == "haa2c_graph" or self.args["algo"] == "hatrpo_graph":
                    self.insertGraph(data)  # insert data into buffer
                else:
                    self.insert(data)  # insert data into buffer

            # print(f"sample spend {time.time() - sample_time}")

            # compute return and update network
            self.compute()
            self.prep_training()  # change to train mode

            train_time = time.time()
            actor_train_infos, critic_train_info = self.train(episode=episode)
            # print(f"train spend {time.time() - train_time}")

            # log information
            if episode % self.algo_args["train"]["log_interval"] == 0:
                self.logger.episode_log(
                    actor_train_infos,
                    critic_train_info,
                    self.actor_buffer,
                    self.critic_buffer,
                )

            # eval
            if episode % self.algo_args["train"]["eval_interval"] == 0:
                if self.algo_args["eval"]["use_eval"]:
                    self.prep_rollout()
                    self.eval()
                self.save()

            self.after_update()

    def warmup(self):
        """Warm up the replay buffer."""
        # reset env
        obs, share_obs, available_actions = self.envs.reset()
        # replay buffer
        for agent_id in range(self.num_agents):
            self.actor_buffer[agent_id].obs[0] = obs[:, agent_id].copy()
            if self.actor_buffer[agent_id].available_actions is not None:
                self.actor_buffer[agent_id].available_actions[0] = available_actions[
                                                                   :, agent_id
                                                                   ].copy()
        if self.state_type == "EP":
            self.critic_buffer.share_obs[0] = share_obs[:, 0].copy()
        elif self.state_type == "FP":
            self.critic_buffer.share_obs[0] = share_obs.copy()

    @torch.no_grad()
    def collect(self, step):
        """Collect actions and values from actors and critics.
        Args:
            step: step in the episode.
        Returns:
            values, actions, action_log_probs, rnn_states, rnn_states_critic
        """
        # collect actions, action_log_probs, rnn_states from n actors
        action_collector = []
        action_log_prob_collector = []
        rnn_state_collector = []
        for agent_id in range(self.num_agents):
            action, action_log_prob, rnn_state = self.actor[agent_id].get_actions(
                self.actor_buffer[agent_id].obs[step],
                self.actor_buffer[agent_id].rnn_states[step],
                self.actor_buffer[agent_id].masks[step],
                self.actor_buffer[agent_id].available_actions[step]
                if self.actor_buffer[agent_id].available_actions is not None
                else None
            )
            action_collector.append(_t2n(action))
            action_log_prob_collector.append(_t2n(action_log_prob))
            rnn_state_collector.append(_t2n(rnn_state))
        # (n_agents, n_threads, dim) -> (n_threads, n_agents, dim)
        actions = np.array(action_collector).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_prob_collector).transpose(1, 0, 2)
        rnn_states = np.array(rnn_state_collector).transpose(1, 0, 2, 3)

        # collect values, rnn_states_critic from 1 critic
        if self.state_type == "EP":
            value, rnn_state_critic = self.critic.get_values(
                self.critic_buffer.share_obs[step],
                self.critic_buffer.rnn_states_critic[step],
                self.critic_buffer.masks[step],
            )
            # (n_threads, dim)
            values = _t2n(value)
            rnn_states_critic = _t2n(rnn_state_critic)
        elif self.state_type == "FP":
            value, rnn_state_critic = self.critic.get_values(
                np.concatenate(self.critic_buffer.share_obs[step]),
                np.concatenate(self.critic_buffer.rnn_states_critic[step]),
                np.concatenate(self.critic_buffer.masks[step]),
            )  # concatenate (n_threads, n_agents, dim) into (n_threads * n_agents, dim)
            # split (n_threads * n_agents, dim) into (n_threads, n_agents, dim)
            values = np.array(
                np.split(_t2n(value), self.algo_args["train"]["n_rollout_threads"])
            )
            rnn_states_critic = np.array(
                np.split(
                    _t2n(rnn_state_critic), self.algo_args["train"]["n_rollout_threads"]
                )
            )

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    @torch.no_grad()
    def collectGraph(self, step):
        """Collect actions and values from actors and critics.
        Args:
            step: step in the episode.
        Returns:
            values, actions, action_log_probs, rnn_states, rnn_states_critic
        """
        # collect actions, action_log_probs, rnn_states from n actors

        action_collector = []
        action_log_prob_collector = []
        rnn_state_collector = []
        father_actions_collector = []
        last_actions, last_actions_collector = [], []
        all_obs = []
        all_actions = []

        for id in range(self.num_agents):
            last_actions.append(self.actor_buffer[id].actions[step])

        for agent_id in range(self.num_agents):
            all_obs.append(self.actor_buffer[agent_id].obs[step])
            all_actions.append(self.actor_buffer[agent_id].actions[step])

        for agent_id in range(self.num_agents):
            action, action_log_prob, rnn_state, father_actions = self.actor[agent_id].get_actions(
                self.actor_buffer[agent_id].obs[step],
                self.actor_buffer[agent_id].rnn_states[step],
                self.actor_buffer[agent_id].masks[step],
                self.actor_buffer[agent_id].available_actions[step]
                if self.actor_buffer[agent_id].available_actions is not None
                else None,
                last_actions=self.actor_buffer[agent_id].last_actions[step],
                n_agents=self.num_agents,
                agent_id=agent_id,
                all_obs=all_obs,
                step_reuse=step,
                graph_actor=self.graph_actor if self.algo_args["algo"]["shareGraph"] == True else None,
                all_actions=all_actions
            )

            action_collector.append(_t2n(action))
            last_actions_array = np.array(last_actions)  # 将列表转换为一个单一的 numpy.ndarray
            last_actions_collector.append(_t2n(torch.tensor(last_actions_array).squeeze(-2)))

            action_log_prob_collector.append(_t2n(action_log_prob))
            rnn_state_collector.append(_t2n(rnn_state))
            father_actions_collector.append(_t2n(father_actions))


        # (n_agents, n_threads, dim) -> (n_threads, n_agents, dim)
        actions = np.array(action_collector).transpose(1, 0, 2)
        last_actions_ = np.array(last_actions_collector).transpose(2, 1, 0, 3)
        action_log_probs = np.array(action_log_prob_collector).transpose(1, 0, 2)
        rnn_states = np.array(rnn_state_collector).transpose(1, 0, 2, 3)
        father_actions = np.array(father_actions_collector).transpose(1, 0, 2)

        # import matplotlib.pyplot as plt
        # x0, y0, z0, x1, y1, z1, x2, y2, z2 = [], [], [], [], [], [], [], [], []
        #
        #
        # x0.append(actions[0, 0])
        # y0.append(actions[0, 1])
        # z0.append(actions[0, 2])
        # x1 = actions[1:3, 0]
        # y1 = actions[1:3, 1]
        # z1 = actions[1:3, 2]
        # x2 = actions[3:, 0]
        # y2 = actions[3:, 1]
        # z2 = actions[3:, 2]
        # fig = plt.figure(figsize=(16, 12))
        # ax = plt.axes(projection="3d")
        # # Add x, and y gridlines for the figure
        # ax.grid(b=True, color='blue', linestyle='-.', linewidth=0.5, alpha=0.3)
        # # Creating the color map for the plot
        # my_cmap = plt.get_cmap('hsv')
        # # Creating the 3D plot
        # sctt = ax.scatter3D(x0, y0, z0, c='b', alpha=0.8, marker='.', s=500) #* r
        # sctt = ax.scatter3D(x1, y1, z1, c='b', alpha=0.8, marker='.', s=500) #^ b
        # sctt = ax.scatter3D(x2, y2, z2, c='b', alpha=0.8, marker='.', s=500) #g
        # # display the plot
        # plt.show()
        # plt.savefig(r'C:\Users\pp\WorkFiles\experiment\heterogeneous\baselines\HA+Pareto\HARL-pareto-BigGraph\1.pdf')
        # plt.close()


        # collect values, rnn_states_critic from 1 critic
        if self.state_type == "EP":
            value, rnn_state_critic = self.critic.get_values(
                self.critic_buffer.share_obs[step],
                self.critic_buffer.rnn_states_critic[step],
                self.critic_buffer.masks[step],
            )
            # (n_threads, dim)
            values = _t2n(value)
            rnn_states_critic = _t2n(rnn_state_critic)
        elif self.state_type == "FP":
            value, rnn_state_critic = self.critic.get_values(
                np.concatenate(self.critic_buffer.share_obs[step]),
                np.concatenate(self.critic_buffer.rnn_states_critic[step]),
                np.concatenate(self.critic_buffer.masks[step]),
            )  # concatenate (n_threads, n_agents, dim) into (n_threads * n_agents, dim)
            # split (n_threads * n_agents, dim) into (n_threads, n_agents, dim)
            values = np.array(
                np.split(_t2n(value), self.algo_args["train"]["n_rollout_threads"])
            )
            rnn_states_critic = np.array(
                np.split(
                    _t2n(rnn_state_critic), self.algo_args["train"]["n_rollout_threads"]
                )
            )

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, father_actions, last_actions_

    def insert(self, data):
        """Insert data into buffer."""
        (
            obs,  # (n_threads, n_agents, obs_dim)
            share_obs,  # (n_threads, n_agents, share_obs_dim)
            rewards,  # (n_threads, n_agents, 1)
            dones,  # (n_threads, n_agents)
            infos,  # type: list, shape: (n_threads, n_agents)
            available_actions,  # (n_threads, ) of None or (n_threads, n_agents, action_number)
            values,  # EP: (n_threads, dim), FP: (n_threads, n_agents, dim)
            actions,  # (n_threads, n_agents, action_dim)
            action_log_probs,  # (n_threads, n_agents, action_dim)
            rnn_states,  # (n_threads, n_agents, dim)
            rnn_states_critic,  # EP: (n_threads, dim), FP: (n_threads, n_agents, dim)
        ) = data

        dones_env = np.all(dones, axis=1)  # if all agents are done, then env is done
        rnn_states[
            dones_env == True
            ] = np.zeros(  # if env is done, then reset rnn_state to all zero
            (
                (dones_env == True).sum(),
                self.num_agents,
                self.recurrent_n,
                self.rnn_hidden_size,
            ),
            dtype=np.float32,
        )

        # If env is done, then reset rnn_state_critic to all zero
        if self.state_type == "EP":
            rnn_states_critic[dones_env == True] = np.zeros(
                ((dones_env == True).sum(), self.recurrent_n, self.rnn_hidden_size),
                dtype=np.float32,
            )
        elif self.state_type == "FP":
            rnn_states_critic[dones_env == True] = np.zeros(
                (
                    (dones_env == True).sum(),
                    self.num_agents,
                    self.recurrent_n,
                    self.rnn_hidden_size,
                ),
                dtype=np.float32,
            )

        # masks use 0 to mask out threads that just finish.
        # this is used for denoting at which point should rnn state be reset
        masks = np.ones(
            (self.algo_args["train"]["n_rollout_threads"], self.num_agents, 1),
            dtype=np.float32,
        )
        masks[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32
        )

        # active_masks use 0 to mask out agents that have died
        active_masks = np.ones(
            (self.algo_args["train"]["n_rollout_threads"], self.num_agents, 1),
            dtype=np.float32,
        )
        active_masks[dones == True] = np.zeros(
            ((dones == True).sum(), 1), dtype=np.float32
        )
        active_masks[dones_env == True] = np.ones(
            ((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32
        )

        # bad_masks use 0 to denote truncation and 1 to denote termination
        if self.state_type == "EP":
            bad_masks = np.array(
                [
                    [0.0]
                    if "bad_transition" in info[0].keys()
                       and info[0]["bad_transition"] == True
                    else [1.0]
                    for info in infos
                ]
            )
        elif self.state_type == "FP":
            bad_masks = np.array(
                [
                    [
                        [0.0]
                        if "bad_transition" in info[agent_id].keys()
                           and info[agent_id]["bad_transition"] == True
                        else [1.0]
                        for agent_id in range(self.num_agents)
                    ]
                    for info in infos
                ]
            )

        for agent_id in range(self.num_agents):
            self.actor_buffer[agent_id].insert(
                obs[:, agent_id],
                rnn_states[:, agent_id],
                actions[:, agent_id],
                action_log_probs[:, agent_id],
                masks[:, agent_id],
                active_masks[:, agent_id],
                available_actions[:, agent_id]
                if available_actions[0] is not None
                else None,
            )

        if self.state_type == "EP":
            self.critic_buffer.insert(
                share_obs[:, 0],
                rnn_states_critic,
                values,
                rewards[:, 0],
                masks[:, 0],
                bad_masks,
            )
        elif self.state_type == "FP":
            self.critic_buffer.insert(
                share_obs, rnn_states_critic, values, rewards, masks, bad_masks
            )

    def insertGraph(self, data):
        """Insert data into buffer."""
        (
            obs,  # (n_threads, n_agents, obs_dim)
            share_obs,  # (n_threads, n_agents, share_obs_dim)
            rewards,  # (n_threads, n_agents, 1)
            dones,  # (n_threads, n_agents)
            infos,  # type: list, shape: (n_threads, n_agents)
            available_actions,  # (n_threads, ) of None or (n_threads, n_agents, action_number)
            values,  # EP: (n_threads, dim), FP: (n_threads, n_agents, dim)
            actions,  # (n_threads, n_agents, action_dim)
            action_log_probs,  # (n_threads, n_agents, action_dim)
            rnn_states,  # (n_threads, n_agents, dim)
            rnn_states_critic,  # EP: (n_threads, dim), FP: (n_threads, n_agents, dim)
            father_actions,
            last_actions,
        ) = data

        dones_env = np.all(dones, axis=1)  # if all agents are done, then env is done
        rnn_states[
            dones_env == True
            ] = np.zeros(  # if env is done, then reset rnn_state to all zero
            (
                (dones_env == True).sum(),
                self.num_agents,
                self.recurrent_n,
                self.rnn_hidden_size,
            ),
            dtype=np.float32,
        )

        # If env is done, then reset rnn_state_critic to all zero
        if self.state_type == "EP":
            rnn_states_critic[dones_env == True] = np.zeros(
                ((dones_env == True).sum(), self.recurrent_n, self.rnn_hidden_size),
                dtype=np.float32,
            )
        elif self.state_type == "FP":
            rnn_states_critic[dones_env == True] = np.zeros(
                (
                    (dones_env == True).sum(),
                    self.num_agents,
                    self.recurrent_n,
                    self.rnn_hidden_size,
                ),
                dtype=np.float32,
            )

        # masks use 0 to mask out threads that just finish.
        # this is used for denoting at which point should rnn state be reset
        masks = np.ones(
            (self.algo_args["train"]["n_rollout_threads"], self.num_agents, 1),
            dtype=np.float32,
        )
        masks[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32
        )

        # active_masks use 0 to mask out agents that have died
        active_masks = np.ones(
            (self.algo_args["train"]["n_rollout_threads"], self.num_agents, 1),
            dtype=np.float32,
        )
        active_masks[dones == True] = np.zeros(
            ((dones == True).sum(), 1), dtype=np.float32
        )
        active_masks[dones_env == True] = np.ones(
            ((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32
        )

        # bad_masks use 0 to denote truncation and 1 to denote termination
        if self.state_type == "EP":
            bad_masks = np.array(
                [
                    [0.0]
                    if "bad_transition" in info[0].keys()
                       and info[0]["bad_transition"] == True
                    else [1.0]
                    for info in infos
                ]
            )
        elif self.state_type == "FP":
            bad_masks = np.array(
                [
                    [
                        [0.0]
                        if "bad_transition" in info[agent_id].keys()
                           and info[agent_id]["bad_transition"] == True
                        else [1.0]
                        for agent_id in range(self.num_agents)
                    ]
                    for info in infos
                ]
            )

        for agent_id in range(self.num_agents):
            self.actor_buffer[agent_id].insert(
                obs[:, agent_id],
                rnn_states[:, agent_id],
                actions[:, agent_id],
                action_log_probs[:, agent_id],
                masks[:, agent_id],
                active_masks[:, agent_id],
                available_actions[:, agent_id]
                if available_actions[0] is not None
                else None,
                father_actions[:, agent_id],
                last_actions[:, agent_id],
            )

        if self.state_type == "EP":
            self.critic_buffer.insert(
                share_obs[:, 0],
                rnn_states_critic,
                values,
                rewards[:, 0],
                masks[:, 0],
                bad_masks,
            )
        elif self.state_type == "FP":
            self.critic_buffer.insert(
                share_obs, rnn_states_critic, values, rewards, masks, bad_masks
            )

    @torch.no_grad()
    def compute(self):
        """Compute returns and advantages.
        Compute critic evaluation of the last state,
        and then let buffer compute returns, which will be used during training.
        """
        if self.state_type == "EP":
            next_value, _ = self.critic.get_values(
                self.critic_buffer.share_obs[-1],
                self.critic_buffer.rnn_states_critic[-1],
                self.critic_buffer.masks[-1],
            )
            next_value = _t2n(next_value)
        elif self.state_type == "FP":
            next_value, _ = self.critic.get_values(
                np.concatenate(self.critic_buffer.share_obs[-1]),
                np.concatenate(self.critic_buffer.rnn_states_critic[-1]),
                np.concatenate(self.critic_buffer.masks[-1]),
            )
            next_value = np.array(
                np.split(_t2n(next_value), self.algo_args["train"]["n_rollout_threads"])
            )
        self.critic_buffer.compute_returns(next_value, self.value_normalizer)

    def train(self, episode=0):
        """Train the model."""
        raise NotImplementedError

    def after_update(self):
        """Do the necessary data operations after an update.
        After an update, copy the data at the last step to the first position of the buffer.
        This will be used for then generating new actions.
        """
        for agent_id in range(self.num_agents):
            self.actor_buffer[agent_id].after_update()
        self.critic_buffer.after_update()

    @torch.no_grad()
    def eval(self):
        """Evaluate the model."""
        self.logger.eval_init()  # logger callback at the beginning of evaluation
        eval_episode = 0

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

        eval_rnn_states = np.zeros(
            (
                self.algo_args["eval"]["n_eval_rollout_threads"],
                self.num_agents,
                self.recurrent_n,
                self.rnn_hidden_size,
            ),
            dtype=np.float32,
        )
        eval_masks = np.ones(
            (self.algo_args["eval"]["n_eval_rollout_threads"], self.num_agents, 1),
            dtype=np.float32,
        )
        step_ = 0
        while True:
            eval_actions_collector = []
            for agent_id in range(self.num_agents):

                if self.args["algo"] == "gmosl" or self.args["algo"] == "happo_graph" or self.args[
                    "algo"] == "haa2c_graph" or self.args["algo"] == "hatrpo_graph":
                    eval_actions, temp_rnn_state = self.actor[agent_id].act(
                        eval_obs[:, agent_id],
                        eval_rnn_states[:, agent_id],
                        eval_masks[:, agent_id],
                        eval_available_actions[:, agent_id]
                        if eval_available_actions[0] is not None
                        else None,
                        deterministic=True
                    )
                    eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                    eval_actions_collector.append(_t2n(eval_actions))
                else:
                    eval_actions, temp_rnn_state = self.actor[agent_id].act(
                        eval_obs[:, agent_id],
                        eval_rnn_states[:, agent_id],
                        eval_masks[:, agent_id],
                        eval_available_actions[:, agent_id]
                        if eval_available_actions[0] is not None
                        else None,
                        deterministic=True,
                    )
                    eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                    eval_actions_collector.append(_t2n(eval_actions))

            eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)

            (
                eval_obs,
                eval_share_obs,
                eval_rewards,
                eval_dones,
                eval_infos,
                eval_available_actions,
            ) = self.eval_envs.step(eval_actions)
            eval_data = (
                eval_obs,
                eval_share_obs,
                eval_rewards,
                eval_dones,
                eval_infos,
                eval_available_actions,
            )
            self.logger.eval_per_step(
                eval_data
            )  # logger callback at each step of evaluation

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[
                eval_dones_env == True
                ] = np.zeros(  # if env is done, then reset rnn_state to all zero
                (
                    (eval_dones_env == True).sum(),
                    self.num_agents,
                    self.recurrent_n,
                    self.rnn_hidden_size,
                ),
                dtype=np.float32,
            )

            eval_masks = np.ones(
                (self.algo_args["eval"]["n_eval_rollout_threads"], self.num_agents, 1),
                dtype=np.float32,
            )
            eval_masks[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32
            )

            step_ += 1
            for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    self.logger.eval_thread_done(
                        eval_i
                    )  # logger callback when an episode is done

            if eval_episode >= self.algo_args["eval"]["eval_episodes"]:
                self.logger.eval_log(
                    eval_episode
                )  # logger callback at the end of evaluation
                break

    @torch.no_grad()
    def render(self):
        """Render the model."""
        print("start rendering")
        if self.manual_expand_dims:
            # this env needs manual expansion of the num_of_parallel_envs dimension
            for _ in range(self.algo_args["render"]["render_episodes"]):
                eval_obs, _, eval_available_actions = self.envs.reset()
                eval_obs = np.expand_dims(np.array(eval_obs), axis=0)
                eval_available_actions = (
                    np.expand_dims(np.array(eval_available_actions), axis=0)
                    if eval_available_actions is not None
                    else None
                )
                eval_rnn_states = np.zeros(
                    (
                        self.env_num,
                        self.num_agents,
                        self.recurrent_n,
                        self.rnn_hidden_size,
                    ),
                    dtype=np.float32,
                )
                eval_masks = np.ones(
                    (self.env_num, self.num_agents, 1), dtype=np.float32
                )
                rewards = 0
                while True:
                    eval_actions_collector = []
                    for agent_id in range(self.num_agents):
                        eval_actions, temp_rnn_state = self.actor[agent_id].act(
                            eval_obs[:, agent_id],
                            eval_rnn_states[:, agent_id],
                            eval_masks[:, agent_id],
                            eval_available_actions[:, agent_id]
                            if eval_available_actions is not None
                            else None,
                            deterministic=True,
                        )
                        eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                        eval_actions_collector.append(_t2n(eval_actions))
                    eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)
                    (
                        eval_obs,
                        _,
                        eval_rewards,
                        eval_dones,
                        _,
                        eval_available_actions,
                    ) = self.envs.step(eval_actions[0])
                    rewards += eval_rewards[0][0]
                    eval_obs = np.expand_dims(np.array(eval_obs), axis=0)
                    eval_available_actions = (
                        np.expand_dims(np.array(eval_available_actions), axis=0)
                        if eval_available_actions is not None
                        else None
                    )
                    if self.manual_render:
                        self.envs.render()
                    if self.manual_delay:
                        time.sleep(0.1)
                    if eval_dones[0]:
                        print(f"total reward of this episode: {rewards}")
                        break
        else:
            # this env does not need manual expansion of the num_of_parallel_envs dimension
            # such as dexhands, which instantiates a parallel env of 64 pair of hands
            for _ in range(self.algo_args["render"]["render_episodes"]):
                eval_obs, _, eval_available_actions = self.envs.reset()
                eval_rnn_states = np.zeros(
                    (
                        self.env_num,
                        self.num_agents,
                        self.recurrent_n,
                        self.rnn_hidden_size,
                    ),
                    dtype=np.float32,
                )
                eval_masks = np.ones(
                    (self.env_num, self.num_agents, 1), dtype=np.float32
                )
                rewards = 0
                while True:
                    eval_actions_collector = []
                    for agent_id in range(self.num_agents):
                        eval_actions, temp_rnn_state = self.actor[agent_id].act(
                            eval_obs[:, agent_id],
                            eval_rnn_states[:, agent_id],
                            eval_masks[:, agent_id],
                            eval_available_actions[:, agent_id]
                            if eval_available_actions[0] is not None
                            else None,
                            deterministic=True,
                        )
                        eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                        eval_actions_collector.append(_t2n(eval_actions))
                    eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)
                    (
                        eval_obs,
                        _,
                        eval_rewards,
                        eval_dones,
                        _,
                        eval_available_actions,
                    ) = self.envs.step(eval_actions)
                    rewards += eval_rewards[0][0][0]
                    if self.manual_render:
                        self.envs.render()
                    if self.manual_delay:
                        time.sleep(0.1)
                    if eval_dones[0][0]:
                        print(f"total reward of this episode: {rewards}")
                        break
        if "smac" in self.args["env"]:  # replay for smac, no rendering
            if "v2" in self.args["env"]:
                self.envs.env.save_replay()
            else:
                self.envs.save_replay()

    def prep_rollout(self):
        """Prepare for rollout."""
        for agent_id in range(self.num_agents):
            self.actor[agent_id].prep_rollout()
        self.critic.prep_rollout()

    def prep_training(self):
        """Prepare for training."""
        for agent_id in range(self.num_agents):
            self.actor[agent_id].prep_training()
        self.critic.prep_training()

    def save(self):
        """Save model parameters."""
        for agent_id in range(self.num_agents):
            policy_actor = self.actor[agent_id].actor
            torch.save(
                policy_actor.state_dict(),
                str(self.save_dir) + "/actor_agent" + str(agent_id) + ".pt",
            )
        policy_critic = self.critic.critic
        torch.save(
            policy_critic.state_dict(), str(self.save_dir) + "/critic_agent" + ".pt"
        )
        if self.value_normalizer is not None:
            torch.save(
                self.value_normalizer.state_dict(),
                str(self.save_dir) + "/value_normalizer" + ".pt",
            )

    def restore(self):
        """Restore model parameters."""
        for agent_id in range(self.num_agents):
            policy_actor_state_dict = torch.load(
                str(self.algo_args["train"]["model_dir"])
                + "/actor_agent"
                + str(agent_id)
                + ".pt"
            )
            self.actor[agent_id].actor.load_state_dict(policy_actor_state_dict)
        if not self.algo_args["render"]["use_render"]:
            policy_critic_state_dict = torch.load(
                str(self.algo_args["train"]["model_dir"]) + "/critic_agent" + ".pt"
            )
            self.critic.critic.load_state_dict(policy_critic_state_dict)
            if self.value_normalizer is not None:
                value_normalizer_state_dict = torch.load(
                    str(self.algo_args["train"]["model_dir"])
                    + "/value_normalizer"
                    + ".pt"
                )
                self.value_normalizer.load_state_dict(value_normalizer_state_dict)

    def close(self):
        """Close environment, writter, and logger."""
        if self.algo_args["render"]["use_render"]:
            self.envs.close()
        else:
            self.envs.close()
            if self.algo_args["eval"]["use_eval"] and self.eval_envs is not self.envs:
                self.eval_envs.close()
            self.writter.export_scalars_to_json(str(self.log_dir + "/summary.json"))
            self.writter.close()
            self.logger.close()

    def TrainIndGraph(self, all_obs, advantages, all_actions, train_info, episode):
        #####################################单独训练图####################################################################

        epi_roll = self.algo_args["train"]["episode_length"] * self.algo_args["train"]["n_rollout_threads"]

        agent_id_graph = torch.eye(self.num_agents,device=self.tpdv['device']).unsqueeze(0).repeat(epi_roll, 1, 1)
        # self.n_rollout_threads, num_agents, num_agents

        all_obs_ = check(all_obs).to(**self.tpdv).reshape(epi_roll, self.num_agents, -1)
        advantages_ = check(advantages).to(**self.tpdv).reshape(epi_roll, -1)

        last_actions_batch_ = check(all_actions).to(**self.tpdv).reshape(epi_roll, self.num_agents, self.n_actions)


        inputs_graph = torch.cat((all_obs_, agent_id_graph, last_actions_batch_), -1).float()

        # graph_time = time.time()
        encoder_output, samples, mask_scores, entropy, adj_prob, \
        log_softmax_logits_for_rewards, entropy_regularization, sparse_loss = self.graph_actor(inputs_graph)
        # print(f"graph speeded {time.time() - graph_time}")

        loss_graph_actor = -(torch.sum(log_softmax_logits_for_rewards, dim=(1, 2)) * torch.sum(
            advantages_.unsqueeze(1).repeat(1, self.num_agents, 1), dim=(1, 2))).mean()
        if self.algo_args["algo"]["sparse_loss"] == True:
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

        loss_graph_actor.backward(retain_graph=True)

        if self.use_max_grad_norm:
            graph_actor_grad_norm = nn.utils.clip_grad_norm_(self.graph_actor.parameters(),
                                                             self.max_grad_norm)
        else:
            graph_actor_grad_norm = get_grad_norm(self.graph_actor.parameters())

        train_info["loss_graph_actor"] = loss_graph_actor.mean()
        self.graph_actor_optimizer.step()
        # if episode > 2500:
        #     self.graph_actor_lr_scheduler.step(loss_graph_actor)
        #########################################################################################################

    def lr_decay_graph(self, episode, episodes):
        update_linear_schedule(self.graph_actor_optimizer, episode, episodes, self.lr_graph)
