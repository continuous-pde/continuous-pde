import torch
import numpy as np
import os
from torch.utils.data import Dataset
import pickle
import matplotlib.pyplot as plt


class ShallowWaterDataset(Dataset):
    def __init__(self, mode, n_frames, space_subsampling, time_subsampling=1.0, generalize=False,
                 uniform=False):
        """ Dataset for the Eagle dataset
        :param mode: ["train", "test", "valid"] - determines which data is loaded
        :param n_frames: number of frames per sequences [2, T]
        :param space_subsampling: fraction of points to keep in the spatial domain [0, 1]
        :param time_subsampling: how many frames to remove (1 = keep all, 2 = 1/2 of the frames, etc.)
        """
        super(ShallowWaterDataset, self).__init__()
        assert mode in ['train', 'test', 'valid']
        self.mode = mode if mode != 'valid' else 'train'

        # Choose first path that exists
        path_bases = ["../Data/ShallowWater", "Data/ShallowWater"]
        try:
            path_base = next(path for path in path_bases if os.path.exists(path))
        except StopIteration:
            raise FileNotFoundError("No path found for data among {}".format(path_bases))

        self.fn = path_base
        self.data = pickle.load(open(os.path.join(self.fn, self.mode + '.pkl'), 'rb'))

        self.dt_eval = 1
        self.t_horizon = n_frames * self.dt_eval

        # Create spatial and temporal masks
        torch.manual_seed(0 if not generalize else 1)
        self.space_mask = (torch.rand(128, 64) >= 1.0 - space_subsampling).view(-1)
        if uniform:
            self.temporal_mask = torch.zeros(n_frames).bool()
            self.temporal_mask[::int(time_subsampling)] = True
            self.temporal_mask[0] = True
        else:
            self.temporal_mask = (torch.rand(n_frames) >= 1 - time_subsampling).view(-1)
            self.temporal_mask[0] = True
            self.temporal_mask[-1] = True

        # Boring piece of code to compute how many sequences we can extract in each full simulation
        total_traj_length = len(self.data[0]['data'])
        assert n_frames <= total_traj_length, "n_frames must be less than total_traj_length, but {} > {}".format(
            n_frames,
            total_traj_length)
        self.seq_length = int(self.t_horizon / self.dt_eval)
        self.n_seq_per_traj = int(total_traj_length / self.seq_length)
        self.dataset_size = self.n_seq_per_traj * len(self.data)

        self.state_dim = 2
        self.coord_dim = 2

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        traj_id = index // self.n_seq_per_traj
        seq_id = index % self.n_seq_per_traj

        data = self.data[traj_id]
        x = data['data'].float()

        out = {'space_mask': self.space_mask,
               'time_mask': self.temporal_mask,
               'index': index,
               'coords_euclidean': data['coords'].float(),
               'coords': data['coords_ang'].float()}

        start = seq_id * self.seq_length
        stop = (seq_id + 1) * self.seq_length
        out['data'] = x[start:stop]
        out['t'] = torch.arange(self.seq_length).float() * self.dt_eval

        out['data'] = out['data'].reshape(self.seq_length, -1, self.state_dim)
        out['coords'] = out['coords'].reshape(-1, self.coord_dim)
        out['node_type'] = torch.zeros_like(out['coords'][..., 0]).long()
        return out


def generate_skipped_lat_lon_mask(coords, base_jump=1):
    lons = coords[:, 0, 0].cpu().numpy()
    lats = coords[0, :, 1].cpu().numpy()
    n_lon = lons.size
    delta_dis_equator = 2 * np.pi * 1 / n_lon
    mask_list = []
    for lat in lats:
        delta_dis_lat = 2 * np.pi * np.sin(lat) / n_lon
        ratio = delta_dis_lat / delta_dis_equator
        n = int(np.ceil(np.log(ratio + 1e-6) / np.log(2 / 5)))
        mask = torch.zeros(n_lon)
        mask[::2 ** (n - 1 + base_jump)] = 1
        mask_list.append(mask)

    mask = torch.stack(mask_list, dim=-1)
    return mask


if __name__ == '__main__':
    for s in [1, 2, 4]:
        dataset = ShallowWaterDataset('train', n_frames=21, space_subsampling=1, time_subsampling=s)

        from utils.accumulator import Accumulator
        from scipy.interpolate import griddata
        import torch.nn as nn
        from tqdm import tqdm

        accu = Accumulator()
        for x in tqdm(dataset):
            groundtruth = x['data']
            time_mask = x['time_mask']
            space_mask = x['space_mask']
            time = x['t']
            position = x['coords']

            phi = groundtruth[time_mask][:, space_mask]
            t = time[time_mask]
            coords = position[space_mask]

            interp = griddata(t.cpu().numpy(), phi.cpu().numpy(), time.cpu().numpy(), method='cubic')
            time_interp = torch.from_numpy(interp)

            interpolation = []
            for t in range(time_interp.shape[0]):
                interp = griddata(coords.cpu().numpy(), time_interp[t].cpu().numpy(), position.cpu().numpy(),
                                  method='cubic')
                nan_position = np.isnan(interp[..., 0])
                interp[nan_position] = griddata(coords.cpu().numpy(), time_interp[t].cpu().numpy(),
                                                position.cpu().numpy()[nan_position, :], method='nearest')

                interpolation.append(torch.from_numpy(interp))
            interpolation = torch.stack(interpolation, dim=0)

            loss_out_t = nn.MSELoss()(interpolation[~time_mask], groundtruth[~time_mask])
            loss_in_t = nn.MSELoss()(interpolation[time_mask], groundtruth[time_mask])
            loss_in_t_in_s = nn.MSELoss()(interpolation[time_mask][:, space_mask],
                                          groundtruth[time_mask][:, space_mask])
            loss_in_t_out_s = nn.MSELoss()(interpolation[time_mask][:, ~space_mask],
                                           groundtruth[time_mask][:, ~space_mask])
            loss_out_t_in_s = nn.MSELoss()(interpolation[~time_mask][:, space_mask],
                                           groundtruth[~time_mask][:, space_mask])
            loss_out_t_out_s = nn.MSELoss()(interpolation[~time_mask][:, ~space_mask],
                                            groundtruth[~time_mask][:, ~space_mask])

            accu.add({'loss_out_t': loss_out_t.item(),
                      'loss_in_t': loss_in_t.item(),
                      'loss_in_t_in_s': loss_in_t_in_s.item(),
                      'loss_in_t_out_s': loss_in_t_out_s.item(),
                      'loss_out_t_in_s': loss_out_t_in_s.item(),
                      'loss_out_t_out_s': loss_out_t_out_s.item()}, 1)
        print("Space subsampling: {}".format(s))
        print(accu.mean())
        print()
