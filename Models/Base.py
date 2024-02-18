import torch
import torch.nn as nn
from typing import Optional, Union, List
import torch.nn.functional as F
import math
import numpy as np
from torch_scatter import scatter_sum


class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x):
        return (x * torch.sigmoid_(x * F.softplus(self.beta))).div_(1.1)


class MLP(nn.Module):
    def __init__(self, input_size: int, output_size: int, h_size: int, n_layers: int,
                 activation: nn.modules = nn.ReLU, layernorm: bool = False,
                 last_activation: bool = False):
        """ A simple MLP with n_layers hidden layers of size h_size and ReLU activation.
        :param input_size: Size of input
        :param output_size: Size of output
        :param h_size: Size of hidden layers
        :param n_layers: Number of hidden layers
        :param activation: Activation function
        :param layernorm: Whether to use layer normalization
        """
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, h_size))
        if layernorm:
            self.layers.append(nn.LayerNorm(h_size))
        self.layers.append(activation())
        for i in range(n_layers - 1):
            self.layers.append(nn.Linear(h_size, h_size))
            if layernorm:
                self.layers.append(nn.LayerNorm(h_size))
            self.layers.append(activation())
        self.layers.append(nn.Linear(h_size, output_size))
        if last_activation:
            self.layers.append(activation())
        self.input_size = input_size

    def forward(self, x):
        assert x.shape[-1] == self.input_size, "Input size mismatch, expected {}, got {}".format(self.input_size,
                                                                                                 x.shape[-1])
        for layer in self.layers:
            x = layer(x)
        return x


class GNN(nn.Module):
    def __init__(self, n_hidden=2, latent_size=128, with_edge_features=True, node_type_size=0, activation="relu"):
        super(GNN, self).__init__()
        if with_edge_features:
            in_size = latent_size * 3 + 2 * node_type_size
        else:
            in_size = latent_size * 2 + 2 * node_type_size

        activations = {"relu": nn.ReLU, "swish": Swish, "tanh": nn.Tanh}
        self.f_edge = MLP(input_size=in_size, n_layers=n_hidden, layernorm=False,
                          output_size=latent_size, h_size=128, activation=activations[activation])
        self.f_node = MLP(input_size=latent_size + latent_size + node_type_size, n_layers=n_hidden, layernorm=False,
                          output_size=latent_size, h_size=128, activation=activations[activation])

    def forward(self, V, E, edges, batch):
        senders = V[edges[0]]
        receivers = V[edges[1]]
        if E is not None:
            edge_inpt = torch.cat([senders, receivers, E], dim=-1)
        else:
            edge_inpt = torch.cat([senders, receivers], dim=-1)

        edge_embeddings = self.f_edge(edge_inpt)
        edge_sum = scatter_sum(edge_embeddings, edges[0], dim=0)
        node_inpt = torch.cat([V, edge_sum], dim=-1)
        node_embeddings = self.f_node(node_inpt)

        return node_embeddings, edge_embeddings


if __name__ == '__main__':
    pass
