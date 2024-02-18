import os.path
import random
from torch.utils.data import Dataset
from torch.nn.functional import one_hot
import pickle
import torch
import matplotlib.animation as animation
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class EagleDataset(Dataset):
    def __init__(self, mode, n_frames=600, space_subsampling=1.0, time_subsampling=1.0):
        """ Dataset for the Eagle dataset
        :param mode: ["train", "test", "valid"] - determines which data is loaded
        :param n_frames: number of frames per sequences [2, T]
        :param space_subsampling: fraction of points to keep in the spatial domain [0, 1]
        :param time_subsampling: how many frames to remove (1 = keep all, 2 = 1/2 of the frames, etc.)
        """
        super(EagleDataset, self).__init__()
        assert mode in ["train", "test", "valid"]
        self.mode = mode

        PATH = ""
        assert PATH != "", "Please set the path to the Eagle dataset"

        self.fn = PATH
        assert os.path.exists(self.fn)

        self.dt_eval = 1 / 30
        self.t_horizon = n_frames * self.dt_eval

        # Create spatial and temporal masks
        torch.manual_seed(0)
        self.space_subsampling = space_subsampling

        self.temporal_mask = torch.zeros(n_frames).bool()
        self.temporal_mask[::int(time_subsampling)] = True
        self.temporal_mask[0] = True

        self.dataloc = []
        try:
            with open(f"../Datasets/{mode}.txt", "r") as f:
                for line in f.readlines():
                    self.dataloc.append(os.path.join(self.fn, line.strip()))
        except FileNotFoundError:
            with open(f"Datasets/{mode}.txt", "r") as f:
                for line in f.readlines():
                    self.dataloc.append(os.path.join(self.fn, line.strip()))

        self.seq_length = n_frames
        self.state_dim = 4
        self.coord_dim = 2

    def __len__(self):
        return len(self.dataloc)

    def __getitem__(self, item):
        # Randomly sample a timestep in the sequence. Fixed during evaluation for reproducibility
        t = 100 if self.mode != "train" else random.randint(50, 500)
        path = self.dataloc[item]
        data = np.load(os.path.join(path, 'sim.npz'), mmap_mode='r')

        # Load data and crop to sequence length
        mesh_pos = data["pointcloud"][t].copy()
        node_type = data["mask"][t].copy()
        Vx = data['VX'][t:t + self.seq_length].copy()
        Vy = data['VY'][t:t + self.seq_length].copy()
        Ps = data['PS'][t:t + self.seq_length].copy()
        Pg = data['PG'][t:t + self.seq_length].copy()

        velocity = np.stack([Vx, Vy], axis=-1)
        pressure = np.stack([Ps, Pg], axis=-1)

        mesh_pos = torch.from_numpy(mesh_pos)
        node_type = torch.from_numpy(node_type)
        velocity = torch.from_numpy(velocity)
        pressure = torch.from_numpy(pressure)

        # Life is better when data is normalized
        velocity, pressure = self.normalize(velocity, pressure)

        out = {'node_type': node_type.long(),
               'time_mask': self.temporal_mask.bool(),
               'index': int(item),
               'coords': mesh_pos.float(),
               'data': torch.cat([velocity, pressure], dim=-1).float(),
               't': torch.arange(self.seq_length).float() * self.dt_eval, }

        if len(out["coords"]) < 3000:
            delta = 3000 - len(out["coords"])
            out["coords"] = torch.cat([out["coords"], out["coords"][:delta]])
            out["data"] = torch.cat([out["data"], out["data"][:, :delta]], dim=1)
            out["node_type"] = torch.cat([out["node_type"], out["node_type"][:delta]])

        # Points are somewhat sorted, so we shuffle them
        shuffle = torch.randperm(len(out['coords']))
        out['coords'] = out['coords'][shuffle]
        out['data'] = out['data'][:, shuffle]
        out['node_type'] = out['node_type'][shuffle]

        # Crop at 3000 points
        out['coords'] = out['coords'][:3000]
        out['data'] = out['data'][:, :3000]
        out['node_type'] = out['node_type'][:3000]

        # Create the space subsampling mask
        space_mask = torch.zeros_like(out["node_type"]).float()
        space_mask = torch.logical_or((torch.rand_like(space_mask) >= 1 - self.space_subsampling),
                                      (out["node_type"] != 0))

        out["space_mask"] = space_mask.bool()
        out['coords'] = out['coords'] + torch.rand_like(out['coords']) * 0.001

        return out

    def normalize(self, velocity=None, pressure=None):
        if pressure is not None:
            pressure_shape = pressure.shape
            if pressure.shape[-1] == 1:
                mean = torch.tensor([3.9535]).to(pressure.device)
                std = torch.tensor([11.2199]).to(pressure.device)
                pressure = pressure.reshape(-1)
            else:
                mean = torch.tensor([-0.8322, 4.6050]).to(pressure.device)
                std = torch.tensor([7.4013, 9.7232]).to(pressure.device)
                pressure = pressure.reshape(-1, 2)
            pressure = (pressure - mean) / std
            pressure = pressure.reshape(pressure_shape)
        if velocity is not None:
            velocity_shape = velocity.shape
            mean = torch.tensor([-0.0015, 0.2211]).to(velocity.device).view(-1, 2)
            std = torch.tensor([1.7970, 2.0258]).to(velocity.device).view(-1, 2)
            velocity = velocity.reshape(-1, 2)
            velocity = (velocity - mean) / std
            velocity = velocity.reshape(velocity_shape)

        return velocity, pressure

    def denormalize(self, velocity=None, pressure=None):
        if pressure is not None:
            pressure_shape = pressure.shape
            if pressure.shape[-1] == 1:
                mean = torch.tensor([3.9535]).to(pressure.device)
                std = torch.tensor([11.2199]).to(pressure.device)
                pressure = pressure.reshape(-1)
            else:
                mean = torch.tensor([-0.8322, 4.6050]).to(pressure.device)
                std = torch.tensor([7.4013, 9.7232]).to(pressure.device)
                pressure = pressure.reshape(-1, 2)
            pressure = (pressure * std) + mean
            pressure = pressure.reshape(pressure_shape)
        if velocity is not None:
            velocity_shape = velocity.shape
            mean = torch.tensor([-0.0015, 0.2211]).to(velocity.device).view(-1, 2)
            std = torch.tensor([1.7970, 2.0258]).to(velocity.device).view(-1, 2)
            velocity = velocity.reshape(-1, 2)
            velocity = velocity * std + mean
            velocity = velocity.reshape(velocity_shape)

        return velocity, pressure


if __name__ == '__main__':
    for s in [2, 4]:
        dataset = EagleDataset('test', n_frames=21, space_subsampling=0.25, time_subsampling=s)

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
            phi += torch.randn_like(phi)
            t = time[time_mask]
            coords = position[space_mask]

            interp = griddata(t.cpu().numpy(), phi.cpu().numpy(), time.cpu().numpy(), method='cubic')
            time_interp = torch.from_numpy(interp)

            interpolation = []
            for t in range(time_interp.shape[0]):
                interp = griddata(coords.cpu().numpy(), time_interp[t].cpu().numpy(), position.cpu().numpy(),
                                  method='linear')
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
        print("space_subsampling: {}".format(s))
        print(accu.mean())
        print()
