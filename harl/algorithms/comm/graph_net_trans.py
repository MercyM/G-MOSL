import torch
import torch.nn as nn
import torch.nn.functional as f
from harl.algorithms.comm.util import check
from torch_geometric.nn import GATConv
import itertools
from harl.models.base.distributions import FixedCategorical


class TransEncoder(nn.Module):
    def __init__(self, n_xdims, nhead, num_layers):
        super(TransEncoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(n_xdims, nhead, 128)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)

    def forward(self, inputs):
        out = self.transformer_encoder(inputs)
        return out


class GATEncoder(nn.Module):
    def __init__(self, args, n_xdims, gat_nhead, node_num, n_actions, num_agents, device=torch.device("cpu")):
        super(GATEncoder, self).__init__()
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.node_num = node_num
        self.GAT1 = GATConv(n_xdims, n_actions * num_agents, heads=gat_nhead, concat=True, dropout=0.3)  # 0.3
        self.GAT2 = GATConv(n_actions * num_agents * gat_nhead, n_xdims, dropout=0.3)  # 0.3
        self.args = args
        self.to(device)
        self.edge_index = None

    def cuda_transfer(self):
        self.GAT1.cuda()
        self.GAT2.cuda()

    def gen_edge_index_fc(self, number):
        tmp_lst = list(itertools.permutations(range(0, number), 2))
        edge_index_full_connect = torch.Tensor([list(i) for i in tmp_lst]).t().long()
        return edge_index_full_connect

    def forward(self, x):
        # self.cuda_transfer()
        self.edge_index = self.gen_edge_index_fc(self.node_num).cuda()
        if (len(x.shape) == 2):
            x = f.relu(self.GAT1(x, self.edge_index))
            x = self.GAT2(x, self.edge_index)
        elif (len(x.shape) == 3):
            out_list = []
            for i in range(x.shape[0]):
                x_ = f.relu(self.GAT1(x[i], self.edge_index))
                x_ = self.GAT2(x_, self.edge_index)
                out_list.append(x_)
            x = torch.stack(out_list)
        else:
            print("shape == 4 !!!!!!!!!!!!!!!!!!!!!!!!")
        # return F.log_softmax(x, dim=1)
        return x


from torch_geometric.nn import SAGEConv


class GNNEncoder(nn.Module):
    def __init__(self, args, n_xdims, node_num, n_actions, num_agents, device=torch.device("cpu")):
        super(GNNEncoder, self).__init__()
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.node_num = node_num
        self.conv1 = SAGEConv(n_xdims, n_actions * num_agents)
        self.conv2 = SAGEConv(n_actions * num_agents, n_xdims)
        self.args = args
        self.to(device)
        self.edge_index = None

    def gen_edge_index_fc(self, number):
        tmp_lst = list(itertools.permutations(range(0, number), 2))
        edge_index_full_connect = torch.Tensor([list(i) for i in tmp_lst]).t().long()
        return edge_index_full_connect

    def forward(self, x):
        # self.cuda_transfer()
        self.edge_index = self.gen_edge_index_fc(self.node_num).cuda()
        if (len(x.shape) == 2):
            x = f.relu(self.conv1(x, self.edge_index))
            x = self.conv2(x, self.edge_index)
        elif (len(x.shape) == 3):
            out_list = []
            for i in range(x.shape[0]):
                x_ = f.relu(self.conv1(x[i], self.edge_index))
                x_ = self.conv2(x_, self.edge_index)
                out_list.append(x_)
            x = torch.stack(out_list)
        else:
            print("shape == 4 !!!!!!!!!!!!!!!!!!!!!!!!")
        # return F.log_softmax(x, dim=1)
        return x


class SingleLayerDecoder(nn.Module):
    def __init__(self, args, n_xdims, obs_shape, num_agents, decoder_hidden_dim, node_num, device=torch.device("cpu")):
        super(SingleLayerDecoder, self).__init__()
        self.args = args

        self.max_length = node_num
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.fc_l = nn.Linear(n_xdims, decoder_hidden_dim, bias=True)
        self.fc_r = nn.Linear(n_xdims, decoder_hidden_dim, bias=True)
        self.fc_3 = nn.Linear(decoder_hidden_dim, 1, bias=True)
        # self.fc_l = nn.Linear(n_xdims, decoder_hidden_dim)
        # self.fc_r = nn.Linear(n_xdims, decoder_hidden_dim)
        # self.fc_3 = nn.Linear(decoder_hidden_dim, 1)
        # self.tanh_ = f.tanh()

        self.init_weights()

        self.n_xdims = n_xdims
        self.num_agents = num_agents

        self.to(device)

        self.samples = []
        self.mask_scores = []
        self.entropy = []

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input, ):

        dot_l = self.fc_l(input)  # bz, num, dim
        dot_r = self.fc_r(input)
        tiled_l = dot_l.unsqueeze(2).repeat(1, 1, dot_l.shape[1], 1)  # bz, num, num, dim
        tiled_r = dot_r.unsqueeze(1).repeat(1, dot_r.shape[1], 1, 1)
        # final_sum = torch.tanh(tiled_l + tiled_r)
        # final_sum = tiled_l + tiled_r
        final_sum = torch.relu(tiled_l + tiled_r)
        logits = torch.sigmoid(self.fc_3(final_sum).squeeze(-1))

        self.adj_prob = logits.clone()  # probs input probability, logit input log_probability # 64.12

        # 稀疏性约束 (L1 正则化)
        if self.args["sparse_loss"] == True:
            sparse_loss = 0.01 * torch.sum(torch.abs(logits))
            self.sparse_loss = sparse_loss

        self.samples = []
        self.mask_scores = []
        self.entropy = []

        for i in range(self.max_length):
            position = torch.ones([input.shape[0]]) * i
            position = position.long()

            # Update mask
            self.mask = 1 - f.one_hot(position, num_classes=self.max_length)
            self.mask = check(self.mask).to(**self.tpdv)

            masked_score = self.adj_prob[:, i, :] * self.mask  # logit : input log_probability # 64.12

            prob = torch.distributions.Bernoulli(masked_score)
            # prob = FixedCategorical(masked_score)
            sampled_arr = prob.sample()
            self.samples.append(sampled_arr)
            self.mask_scores.append(masked_score)
            # self.exp_mask_scores.append(torch.exp(masked_score))
            self.entropy.append(prob.entropy())

        if self.args["sparse_loss"] == True:
            return self.samples, self.mask_scores, self.entropy, self.adj_prob, self.sparse_loss
        else:
            return self.samples, self.mask_scores, self.entropy, self.adj_prob, self.adj_prob


class Actor_graph(nn.Module):
    def __init__(self, args, obs_shape, n_actions, num_agents, device=torch.device("cpu")):
        super(Actor_graph, self).__init__()
        self.n_xdims = obs_shape + n_actions + num_agents
        # self.nhead = args["nhead"]
        # self.num_layers = args["num_layers"]
        self.decoder_hidden_dim = args["decoder_hidden_dim"]
        self.node_num = num_agents
        self.gat_nhead = 1  # 1
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.encoder = TransEncoder(self.n_xdims, self.gat_nhead, 2)
        # self.encoder = GATEncoder(args, self.n_xdims, self.gat_nhead, self.node_num, n_actions, num_agents,
        #                           device=device)
        # self.encoder = GNNEncoder(args, self.n_xdims, self.node_num, n_actions, num_agents,device=device)
        self.decoder = SingleLayerDecoder(args, self.n_xdims, obs_shape, num_agents, self.decoder_hidden_dim,
                                          self.node_num,
                                          device=device)

        self.args = args

        self.to(device)

    def forward(self, src):

        import time
        # encode_time=time.time()
        encoder_output = self.encoder(src)
        # print(f"encode speeded {time.time()-encode_time}")

        samples, mask_scores, entropy, adj_prob, sparse_loss = self.decoder(encoder_output)
        # graphs_gen = torch.stack(samples).permute(1, 0, 2) # 1.3.3
        # graph_batch = torch.mean(graphs_gen, dim=0) # 3.3
        logits_for_rewards = torch.stack(mask_scores).permute(1, 0, 2)  # 1.3.3

        ###########################################################
        if self.args["softmax_dim"] == 0:
            log_softmax_logits_for_rewards = f.log_softmax(logits_for_rewards)
        elif self.args["softmax_dim"] == 1:
            log_softmax_logits_for_rewards = f.log_softmax(logits_for_rewards,
                                                           dim=1)  ############# mamujco:hatrpo mappo: -1 haa2c: 1 happo no dim， mpe: 1。
        else:
            log_softmax_logits_for_rewards = f.log_softmax(logits_for_rewards,
                                                           dim=-1)
        ###########################################################
        # log_prob_for_rewards = torch.log(adj_prob) * (1-torch.eye(self.node_num)) # 1.3.3

        entropy_for_rewards = torch.stack(entropy).permute(1, 0, 2)  # 1.3.3
        entropy_regularization = torch.mean(entropy_for_rewards, dim=[1, 2])

        samples = torch.stack(samples).permute(1, 0, 2)
        mask_scores = torch.stack(mask_scores).permute(1, 0, 2)
        entropy = torch.stack(entropy).permute(1, 0, 2)

        return encoder_output, samples, mask_scores, entropy, adj_prob, log_softmax_logits_for_rewards, entropy_regularization, sparse_loss


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.n_xdims = 8
    args.nhead = 4
    args.num_layers = 2
    args.decoder_hidden_dim = 16
    args.node_num = 3
    args.critic_hidden_dim = 16

    src = torch.rand(5, 3, 8)

    actor = Actor_graph(args)
    encoder_output, samples, mask_scores, entropy, \
    log_softmax_logits_for_rewards, entropy_regularization = actor(src)
