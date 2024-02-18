import torch
import numpy as np
import os
from torch.utils.data import Dataset
import pickle
import matplotlib.pyplot as plt

NODE_NORMAL = 0
NODE_INPUT = 4
NODE_OUTPUT = 5
NODE_WALL = 6
NODE_DISABLE = 2


class NavierStokesDataset(Dataset):
    def __init__(self, mode, n_frames, space_subsampling=1.0, time_subsampling=0.75, uniform=False):
        """ Dataset for the Navier dataset
        :param mode: ["train", "test", "valid"] - determines which data is loaded
        :param n_frames: number of frames per sequences [2, T]
        :param space_subsampling: fraction of points to keep in the spatial domain [0, 1]
        :param time_subsampling: how many frames to remove (1 = keep all, 2 = 1/2 of the frames, etc.)
        """
        super(NavierStokesDataset, self).__init__()
        assert mode in ['train', 'test', 'valid']
        self.mode = mode

        # Choose first path that exists in the list
        path_bases = ["../Data/NavierStokes", "Data/NavierStokes"]
        try:
            path_base = next(path for path in path_bases if os.path.exists(path))
        except StopIteration:
            raise FileNotFoundError("No path found for data among {}".format(path_bases))

        self.fn = path_base

        # Load data
        self.data = pickle.load(open(os.path.join(self.fn, self.mode + '.pkl'), 'rb'))

        self.dt_eval = 1
        self.t_horizon = n_frames * self.dt_eval

        # Create spatial and temporal masks
        torch.manual_seed(0)
        self.spatial_mask = (torch.rand(64, 64) >= 1 - space_subsampling).view(-1)
        if uniform:
            self.temporal_mask = torch.zeros(n_frames).bool()
            self.temporal_mask[::int(time_subsampling)] = True
            self.temporal_mask[0] = True
        else:
            self.temporal_mask = (torch.rand(n_frames) >= 1 - time_subsampling).view(-1)
            self.temporal_mask[0] = True
            self.temporal_mask[-1] = True

        # Boring piece of code to compute how many sequences we can extract in each full simulation
        total_traj_length = self.data[0]['data'].shape[1]
        assert n_frames <= total_traj_length, "n_frames must be less than total_traj_length, but {} > {}".format(
            n_frames,
            total_traj_length)
        self.seq_length = int(self.t_horizon / self.dt_eval)
        self.n_seq_per_traj = int(total_traj_length / self.seq_length)
        self.dataset_size = self.n_seq_per_traj * len(self.data)

        self.state_dim = 1
        self.coord_dim = 2
        self.t_resolution = len(self.data[0]['t'])
        self.x_resolution = [64, 64]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        traj_id = index // self.n_seq_per_traj
        seq_id = index % self.n_seq_per_traj

        data = self.data[traj_id]
        coords = data['coords'].float()
        x = data['data'].float()

        out = {'space_mask': self.spatial_mask,
               'time_mask': self.temporal_mask,
               'index': index,
               'coords': coords}

        node_type = torch.ones_like(coords[..., 0]) * NODE_NORMAL
        node_type[0] = NODE_WALL
        node_type[-1] = NODE_WALL
        node_type[:, 0] = NODE_WALL
        node_type[:, -1] = NODE_WALL
        out['node_type'] = node_type.reshape(-1)

        start = seq_id * self.seq_length
        stop = (seq_id + 1) * self.seq_length

        out['data'] = x.permute(1, 2, 3, 0)[start:stop]
        out['t'] = torch.arange(self.seq_length).float() * self.dt_eval

        out['data'] = out['data'].reshape(len(out['t']), -1, self.state_dim)
        out['coords'] = out['coords'].reshape(-1, self.coord_dim)

        return out


if __name__ == '__main__':

    dataset = NavierStokesDataset("test", n_frames=20, space_subsampling=0.25, time_subsampling=1)

    for x in dataset:
        state = x['data']
        pos = x['coords']

        mask = torch.rand_like(pos[..., 0]) < 0.005
        pos = pos[mask]
        pos = torch.rand_like(pos)
        state = state[:, mask, :]

        for t in range(state.shape[0]):
            fig, ax = plt.subplots(figsize=(6, 6))

            ax.scatter(pos[:, 0], pos[:, 1], c=state[t, :, 0], cmap='jet', s=250, marker='s')

            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            plt.show()
