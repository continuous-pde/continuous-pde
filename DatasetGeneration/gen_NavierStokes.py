import torch
import numpy as np
from pde import ScalarField, UnitGrid, MemoryStorage
from pde.pdes import WavePDE
from tqdm import tqdm
import pickle
import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='valid')
args = parser.parse_args()

T_HORIZON = 20
DT = 1
N_SEQ = 32 if args.mode in ["test", "valid"] else 512
SEQ_PER_TRAJ = 2
SIZE = 64


class GaussianRF(object):
    def __init__(self, size, alpha=2., tau=3):
        self.dim = 2
        sigma = tau ** (0.5 * (2 * alpha - self.dim))
        k_max = size // 2
        wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1),
                                torch.arange(start=-k_max, end=0, step=1)), 0).repeat(size, 1)
        k_x = wavenumers.transpose(0, 1)
        k_y = wavenumers
        self.sqrt_eig = (size ** 2) * math.sqrt(2.0) * sigma * (
                (4 * (math.pi ** 2) * (k_x ** 2 + k_y ** 2) + tau ** 2) ** (-alpha / 2.0))
        self.sqrt_eig[0, 0] = 0.0

        self.size = []
        for j in range(self.dim):
            self.size.append(size)
        self.size = tuple(self.size)

    def sample(self):
        coeff = torch.randn(*self.size, dtype=torch.cfloat)
        coeff = self.sqrt_eig * coeff
        u = torch.fft.ifftn(coeff)
        u = u.real
        return u


def navier_stokes_2d(w0, f, visc, T, delta_t, record_steps):
    N = w0.size()[-1]
    k_max = math.floor(N / 2.0)
    steps = math.ceil(T / delta_t)
    w_h = torch.fft.fftn(w0, (N, N))
    f_h = torch.fft.fftn(f, (N, N))
    if len(f_h.size()) < len(w_h.size()):
        f_h = torch.unsqueeze(f_h, 0)
    record_time = math.floor(steps / record_steps)
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=w0.device),
                     torch.arange(start=-k_max, end=0, step=1, device=w0.device)), 0).repeat(N, 1)
    k_x = k_y.transpose(0, 1)
    lap = 4 * (math.pi ** 2) * (k_x ** 2 + k_y ** 2)
    lap[0, 0] = 1.0
    dealias = torch.unsqueeze(
        torch.logical_and(torch.abs(k_y) <= (2.0 / 3.0) * k_max, torch.abs(k_x) <= (2.0 / 3.0) * k_max).float(), 0)
    sol = torch.zeros(*w0.size(), record_steps, 1, device=w0.device, dtype=torch.float)
    sol_t = torch.zeros(record_steps, device=w0.device)
    c = 0
    t = 0.0
    for j in range(steps):
        if j % record_time == 0:
            w = torch.fft.ifftn(w_h, (N, N))
            sol[..., c, 0] = w.real
            sol_t[c] = t
            c += 1
        psi_h = w_h.clone()
        psi_h = psi_h / lap
        q = psi_h.clone()
        temp = q.real.clone()
        q.real = -2 * math.pi * k_y * q.imag
        q.imag = 2 * math.pi * k_y * temp
        q = torch.fft.ifftn(q, (N, N))
        v = psi_h.clone()
        temp = v.real.clone()
        v.real = 2 * math.pi * k_x * v.imag
        v.imag = -2 * math.pi * k_x * temp
        v = torch.fft.ifftn(v, (N, N))
        w_x = w_h.clone()
        temp = w_x.real.clone()
        w_x.real = -2 * math.pi * k_x * w_x.imag
        w_x.imag = 2 * math.pi * k_x * temp
        w_x = torch.fft.ifftn(w_x, (N, N))
        # Partial y of vorticity
        w_y = w_h.clone()
        temp = w_y.real.clone()
        w_y.real = -2 * math.pi * k_y * w_y.imag
        w_y.imag = 2 * math.pi * k_y * temp
        w_y = torch.fft.ifftn(w_y, (N, N))
        # Non-linear term (u.grad(w)): compute in physical space then back to Fourier space
        F_h = torch.fft.fftn(q * w_x + v * w_y, (N, N))
        # Dealias
        F_h = dealias * F_h
        # Cranck-Nicholson update
        w_h = (-delta_t * F_h + delta_t * f_h + (1.0 - 0.5 * delta_t * visc * lap) * w_h) / \
              (1.0 + 0.5 * delta_t * visc * lap)
        # Update real time (used only for recording)
        t += delta_t

    return sol, sol_t


sampler = GaussianRF(SIZE, alpha=2.5, tau=7)

tt = torch.linspace(0, 1, SIZE + 1)[0:-1]
X, Y = torch.meshgrid(tt, tt)
f = 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y)))


def get_initial_condition(index):
    seed = {'train': index, 'test': np.iinfo(np.int32).max - index, 'valid': np.iinfo(np.int32).max//2 - index}
    np.random.seed(seed[args.mode])
    w0 = sampler.sample()

    state, _ = navier_stokes_2d(w0, f=f, visc=1e-3, T=30, delta_t=1e-3, record_steps=20)
    init_cond = state[..., -1, 0].cpu()
    return init_cond


mgrid = torch.stack(torch.meshgrid(
    2 * [torch.linspace(0, 0.5, steps=SIZE)], indexing='ij'), dim=-1)


def generate_trajectory(index):
    assert index < N_SEQ // SEQ_PER_TRAJ
    t = torch.arange(0, T_HORIZON * SEQ_PER_TRAJ + DT, DT).float()
    with torch.no_grad():
        w0 = get_initial_condition(index)
        state, _ = navier_stokes_2d(w0, f=f, visc=1e-3,
                                    T=T_HORIZON * SEQ_PER_TRAJ, delta_t=1e-3,
                                    record_steps=int(T_HORIZON / DT) * SEQ_PER_TRAJ)

    state = state.permute(3, 2, 0, 1)
    return {'data': state,
            't': t,
            'index': index,
            'coords': mgrid}


if __name__ == '__main__':
    dataset = []
    for i in tqdm(range(N_SEQ // SEQ_PER_TRAJ)):
        print("Generating trajectory", i, "of", N_SEQ // SEQ_PER_TRAJ, "...")
        dataset.append(generate_trajectory(i))

    pickle.dump(dataset, open(f"../Data/NavierStokes/{args.mode}.pkl", "wb"))
