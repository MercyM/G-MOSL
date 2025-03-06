import torch
import torch.nn as nn
from harl.utils.envs_tools import check
from harl.models.base.cnn import CNNBaseGraph, CNNBase
from harl.models.base.mlp import MLPBaseGraph, MLPBase
from harl.models.base.rnn import RNNLayer
from harl.models.base.act_graph import ACTGraphLayer
from harl.utils.envs_tools import get_shape_from_obs_space


class StochasticGraphPolicy(nn.Module):
    """Stochastic policy model. Outputs actions given observations."""

    def __init__(self, args, obs_space, action_space, num_agents, device=torch.device("cpu")):
        """Initialize StochasticPolicy model.
        Args:
            args: (dict) arguments containing relevant model information.
            obs_space: (gym.Space) observation space.
            action_space: (gym.Space) action space.
            device: (torch.device) specifies the device to run on (cpu/gpu).        
        """
        super(StochasticGraphPolicy, self).__init__()
        self.hidden_sizes = args["hidden_sizes"]
        self.args = args
        self.gain = args["gain"]
        self.initialization_method = args["initialization_method"]
        self.use_policy_active_masks = args["use_policy_active_masks"]
        self.use_naive_recurrent_policy = args["use_naive_recurrent_policy"]
        self.use_recurrent_policy = args["use_recurrent_policy"]
        self.recurrent_n = args["recurrent_n"]
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.num_agents = num_agents

        obs_shape = get_shape_from_obs_space(obs_space)
        # base = CNNBaseGraph if len(obs_shape) == 3 else MLPBaseGraph
        base = CNNBase if len(obs_shape) == 3 else MLPBase
        self.base = base(args, obs_shape)

        # self.gru = nn.GRUCell(obs_shape[0], obs_shape[0])

        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            self.rnn = RNNLayer(
                self.hidden_sizes[-1],
                self.hidden_sizes[-1],
                self.recurrent_n,
                self.initialization_method,
            )

        self.act = ACTGraphLayer(
            num_agents,
            action_space,
            self.hidden_sizes[-1],
            self.initialization_method,
            self.gain,
            args,
        )

        self.params_ACTGraphLayer = list(self.act.named_parameters())
        self.params_Graphbase = list(self.base.named_parameters())

        self.to(device)

    def forward(self, obs, rnn_states, masks, G_s, available_actions=None, deterministic=False, step_reuse=1):
        """Compute actions from the given inputs.
        Args:
            obs: (np.ndarray / torch.Tensor) observation inputs into network.
            rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
            masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
            available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
            deterministic: (bool) whether to sample from action distribution or return the mode.
        Returns:
            actions: (torch.Tensor) actions to take.
            action_log_probs: (torch.Tensor) log probabilities of taken actions.
            rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        #################################打印参数###############################################
        # print("len(self.params_ACTGraphLayer)")
        # print(len(self.params_ACTGraphLayer))
        #
        # print("len(self.params_Graphbase)")
        # print(len(self.params_Graphbase))
        #################################打印参数###############################################

        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(obs)

        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            actor_features = actor_features.reshape(-1, self.hidden_sizes[-1])
            rnn_states = rnn_states.reshape(-1, 1, self.hidden_sizes[-1])
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)  # 4.64    4.1.64
            actor_features = actor_features.reshape(obs.shape[0], -1, self.hidden_sizes[-1])

        actions, action_log_probs, father_actions_ = self.act(actor_features, G_s, available_actions,
                                                              deterministic, step_reuse=step_reuse)

        return actions, action_log_probs, rnn_states, father_actions_

    def evaluate_actions(
            self, obs, rnn_states, action, masks, available_actions=None, active_masks=None, father_action=None,

    ):
        """Compute action log probability, distribution entropy, and action distribution.
        Args:
            obs: (np.ndarray / torch.Tensor) observation inputs into network.
            rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
            action: (np.ndarray / torch.Tensor) actions whose entropy and log probability to evaluate.
            masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
            available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
            active_masks: (np.ndarray / torch.Tensor) denotes whether an agent is active or dead.
        Returns:
            action_log_probs: (torch.Tensor) log probabilities of the input actions.
            dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
            action_distribution: (torch.distributions) action distribution.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        father_action = check(father_action).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self.base(obs)

        if self.use_naive_recurrent_policy or self.use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        action_log_probs, dist_entropy, action_distribution = self.act.evaluate_actions(
            actor_features,
            action,
            available_actions,
            active_masks=active_masks if self.use_policy_active_masks else None,
            father_action=father_action,
        )

        return action_log_probs, dist_entropy, action_distribution
