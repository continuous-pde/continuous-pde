import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from Models.Base import MLP, Swish, GNN
from matplotlib.tri import Triangulation
from torch_cluster import knn_graph


class GNS(nn.Module):
    def __init__(self, state_size, coord_size, latent_size=128, gnn_density=1, aggregation="rnn", activation=""):
        super(GNS, self).__init__()
        self.state_size = state_size
        self.encode_node = MLP(input_size=state_size + coord_size + 1, output_size=latent_size, h_size=latent_size,
                               n_layers=2, activation=nn.ReLU)
        self.encode_edge = MLP(input_size=coord_size * 2 + 1, output_size=latent_size, h_size=latent_size, n_layers=2,
                               activation=nn.ReLU)

        self.gnns = MultiLayerGNN(latent_size=latent_size, n_hidden=1, n_layers=gnn_density)
        self.prediction_heads = MLP(input_size=latent_size, output_size=state_size, h_size=latent_size, n_layers=2,
                                    activation=Swish if activation == "" else nn.Tanh)
        self.latent_size = latent_size

        self.mhas = AttentionBlock(self.latent_size, 4)

        self.decoder = MLP(input_size=latent_size, output_size=state_size, h_size=latent_size, n_layers=2,
                           activation=Swish if activation == "" else nn.Tanh)
        self.aggregation = aggregation
        self.rnn = nn.GRU(input_size=latent_size, hidden_size=latent_size, num_layers=2, batch_first=True)
        self.pos_enc = FourierPosEnc(coord_size + 1, latent_size)

    def forward(self, phi_0, x_pos, query_points, n_layers, eval_time):
        """
        phi_0: (batch_size, N, state_size)
        x: (batch_size, N, coord_size)
        query_points: (batch_size, M, coord_size+time)
        """
        embeddings = self.compute_embeddings(phi_0, x_pos, n_layers)
        query_result, attention_mask = self.query(embeddings, query_points, x_pos, eval_time)
        predictions = [self.prediction_heads(embeddings[i]).reshape(*phi_0.shape) for i in range(len(embeddings))]
        return self.decoder(query_result), predictions

    def compute_embeddings(self, phi_0, x_pos, n_layers):
        batch = torch.arange(phi_0.shape[0], device=phi_0.device)
        batch = batch[:, None].repeat(1, phi_0.shape[1]).view(-1)

        if x_pos.shape[-1] == 2:
            edge_list = []
            for b in range(phi_0.shape[0]):
                tri = Triangulation(x_pos[b, :, 0].cpu().numpy(), x_pos[b, :, 1].cpu().numpy())
                edges = torch.from_numpy(tri.edges).to(x_pos.device)
                edges = torch.cat([edges, edges.flip(1)], dim=0)
                edges = edges + b * phi_0.shape[1]
                edge_list.append(edges)

            edge_index = torch.cat(edge_list, dim=0).t().long()
        else:  # 3D graph, fall back to nearest neighbors
            edge_index = knn_graph(x_pos.view(-1, x_pos.shape[-1]), k=6, batch=batch).long()
            edge_index = torch.stack([
                torch.cat([edge_index[0], edge_index[1]], dim=0),
                torch.cat([edge_index[1], edge_index[0]], dim=0)
            ], dim=0)

        node_embeddings, edge_embeddings = self.encoder(phi_0, x_pos, edge_index)

        node_embeddings = node_embeddings.view(-1, self.latent_size)
        edge_embeddings = edge_embeddings.view(-1, self.latent_size)

        embeddings = [node_embeddings]
        for _ in range(n_layers):
            last_embeddings = embeddings[-1]
            delta_v, delta_e = self.gnns(last_embeddings, edge_embeddings, edge_index, batch)
            next_embeddings = last_embeddings + delta_v
            edge_embeddings = edge_embeddings + delta_e
            embeddings.append(next_embeddings)
        return embeddings[1:]

    def query(self, embeddings, query_points, x_pos, eval_time):
        B, M, _ = query_points.shape
        query_embeddings = self.pos_enc(query_points)

        outputs, att = [], []

        for i in range(len(embeddings)):
            Q = query_embeddings

            t = torch.ones_like(x_pos[:, :, 0:1]) * eval_time[i]
            pos_embedding = self.pos_enc(torch.cat([x_pos, t], dim=-1))
            K = embeddings[i].view(B, -1, self.latent_size) + pos_embedding

            out, attention = self.mhas(Q, K)
            att.append(attention.detach())
            outputs.append(out)

        delta_query = torch.stack(outputs, dim=2)
        if self.aggregation == "mean":
            delta_query = delta_query.mean(dim=2)
        elif self.aggregation == "max":
            delta_query, _ = delta_query.max(dim=2)
        else:
            delta_query, _ = self.rnn(delta_query.view(B * M, len(embeddings), self.latent_size))
            delta_query = delta_query[:, -1, :].view(B, M, self.latent_size)

        query_embeddings = delta_query
        return query_embeddings, att

    def encoder(self, phi_0, x_pos, edge_index):
        t_pos = torch.zeros_like(x_pos[:, :, 0:1])
        node_embedding = self.encode_node(torch.cat((phi_0, x_pos, t_pos), dim=-1))
        senders = x_pos.view(-1, x_pos.shape[-1])[edge_index[0]]
        receivers = x_pos.view(-1, x_pos.shape[-1])[edge_index[1]]
        delta = (senders - receivers)
        distance = (delta ** 2).sum(dim=-1, keepdim=True)
        edge_embedding = self.encode_edge(torch.cat((senders, receivers, distance), dim=-1))

        return node_embedding, edge_embedding


class MultiLayerGNN(nn.Module):
    def __init__(self, latent_size, n_hidden, n_layers):
        super().__init__()
        self.gnns = nn.ModuleList([GNN(latent_size=latent_size, n_hidden=n_hidden) for _ in range(n_layers)])

    def forward(self, node_embeddings, edge_embeddings, edge_index, batch):
        for gnn in self.gnns[:-1]:
            delta_v, delta_e = gnn(node_embeddings, edge_embeddings, edge_index, batch)
            node_embeddings = node_embeddings + delta_v
            edge_embeddings = edge_embeddings + delta_e

        delta_v, delta_e = self.gnns[-1](node_embeddings, edge_embeddings, edge_index, batch)
        return delta_v, delta_e


class FourierPosEnc(nn.Module):
    def __init__(self, in_size, out_size):
        super(FourierPosEnc, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.w = nn.Parameter(torch.rand(in_size))

    def forward(self, x):
        n_iter = np.ceil(self.out_size / (2 * self.in_size))
        out = []
        for i in range(int(n_iter)):
            out.append(torch.sin(self.w * x * i))
            out.append(torch.cos(self.w * x * i))
        out = torch.cat(out, dim=-1)
        return out[..., :self.out_size]


class RandomFourierPosEnc(nn.Module):
    def __init__(self, in_size, out_size):
        super(RandomFourierPosEnc, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.w = nn.Parameter(torch.randn(1, in_size, int(np.ceil(out_size / 2))))

    def forward(self, x):
        x_shape = x.shape
        x = x.reshape(-1, self.in_size)
        out = torch.cat([torch.cos(x @ self.w), torch.sin(x @ self.w)], dim=-1)
        return out.reshape(*x_shape[:-1], self.out_size)


class AttentionBlock(nn.Module):
    def __init__(self, w_size, n_heads):
        super(AttentionBlock, self).__init__()
        self.ln1 = nn.LayerNorm(w_size)

        self.attention = nn.MultiheadAttention(embed_dim=w_size, num_heads=n_heads, batch_first=True)
        self.mlp = MLP(input_size=w_size, n_layers=1, output_size=w_size, h_size=w_size, layernorm=False,
                       activation=nn.ReLU)

    def forward(self, Q, K, V=None):
        W2, attention = self.attention(Q, K, V if V is not None else K, average_attn_weights=False)
        W3 = Q + W2
        W5 = self.mlp(W3)
        W6 = W3 + W5
        return W6, attention


if __name__ == '__main__':
    model = SplinePosEnc(3, 128)
    for _ in range(100):
        x = torch.rand(2, 64, 3)
        x = (x - x.min()) / (x.max() - x.min())
        y = model(x)
        print(y.shape)
