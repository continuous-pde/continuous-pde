import torch
import numpy as np
import h5py
from tqdm import tqdm
import os
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train')
args = parser.parse_args()

PATH = f"/Users/steevenjanny/Downloads/shallow_water/shallow_water_{args.mode}"

T_HORIZON = 20
DT = 1
N_SEQ = 16 if args.mode == "test" else 64
SEQ_PER_TRAJ = 8
SIZE = (128, 64)


def read_data(index):
    t = torch.arange(0, T_HORIZON, DT).float()
    filename = os.path.join(PATH, f"traj_{str(index).zfill(4)}.h5")
    raw_data = h5py.File(filename, mode='r')

    coords_list = []

    phi = torch.tensor(raw_data['tasks/vorticity'].dims[1][0][:].ravel()[::2])
    theta = torch.tensor(raw_data['tasks/vorticity'].dims[2][0][:].ravel()[::2])

    coords_ang = torch.stack(torch.meshgrid(*[phi, theta], indexing='ij'), dim=-1)
    phi_vert = coords_ang[..., 0]
    theta_vert = coords_ang[..., 1]
    r = 1
    x = torch.cos(phi_vert) * torch.sin(theta_vert) * r
    y = torch.sin(phi_vert) * torch.sin(theta_vert) * r
    z = torch.cos(theta_vert) * r
    coords_list.append(torch.stack([x, y, z], dim=-1))

    coords = torch.cat(coords_list, dim=-1).float()

    data = torch.stack([
        torch.from_numpy(raw_data['tasks/height'][:, ::2, ::2]) * 3000.,
        torch.from_numpy(raw_data['tasks/vorticity'][:, ::2, ::2] * 2)], dim=0)

    data = data.float().permute(1, 2, 3, 0)

    return {'data': data,
            't': t,
            'index': index,
            'coords': coords,
            'coords_ang': coords_ang}


if __name__ == '__main__':
    data = []
    for index in tqdm(range(N_SEQ // SEQ_PER_TRAJ)):
        data.append(read_data(index))

    with open(f"../Data/ShallowWater/{args.mode}.pkl", "wb") as f:
        pickle.dump(data, f)
