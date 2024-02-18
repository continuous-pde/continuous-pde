import torch
from Dataloader.NavierStokesDataset import NavierStokesDataset
from Dataloader.ShallowWaterDataset import ShallowWaterDataset
from Dataloader.EagleDataset import EagleDataset
import random
import numpy as np
from torch.utils.data import DataLoader
from Models.continuous_GNN import GNS
import argparse
from tqdm import tqdm
import pickle
import os
import sys
from utils.accumulator import Accumulator
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=1e-3)

parser.add_argument('--n_frames', type=int, default=21)
parser.add_argument('--dataset', type=str, default='navier')
parser.add_argument('--space_sub', type=float, default=0.25)
parser.add_argument('--time_sub', type=float, default=4)

parser.add_argument('--gnn_subsampling', type=float, default=0.75)
parser.add_argument('--gnn_density', type=int, default=4)
parser.add_argument('--delta', type=int, default=2)

parser.add_argument('--batchsize', type=int, default=16)
parser.add_argument('--n_points', type=int, default=1024)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--save', type=str, default="false")
parser.add_argument('--aggregation', type=str, default="rnn")
parser.add_argument('--activation', type=str, default="")
parser.add_argument('--w_forecast', type=float, default=1)
parser.add_argument('--name', type=str, default='')
args = parser.parse_args()

datasets = {'navier': NavierStokesDataset, 'shallow': ShallowWaterDataset, 'eagle': EagleDataset}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCHSIZE = args.batchsize
SAVE_PATH = f"../trained_models/{args.dataset}"
NUM_WORKER = 4
criterion = nn.MSELoss()


def evaluate():
    print(args)
    torch.random.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    dataset = datasets[args.dataset]
    dataset = dataset(mode='test', n_frames=args.n_frames, space_subsampling=args.space_sub,
                      time_subsampling=args.time_sub, uniform=True)

    loader = DataLoader(dataset, batch_size=BATCHSIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = GNS(state_size=dataset.state_dim, coord_size=dataset.coord_dim,
                latent_size=128, gnn_density=args.gnn_density, aggregation=args.aggregation).to(device)
    try:
        model.load_state_dict(torch.load(f"{SAVE_PATH}/{args.name}.pth", map_location=device))
    except FileNotFoundError:
        model.load_state_dict(torch.load(f"{SAVE_PATH.replace('../', '')}/{args.name}.pth", map_location=device))

    results = validate(model, loader, 0, viz=True, with_stats=True)

    for key in args.__dict__:
        results[key] = args.__dict__[key]

    target_path = f"../results/{args.dataset}" + "_gen"
    target_name = args.name

    os.makedirs(target_path, exist_ok=True)
    with open(f"{target_path}/{target_name}.txt", "w") as f:
        f.write(str(results).replace(', ', ',\n'))


T_training = 10 if args.dataset == "eagle" else 20


def validate(model, valid_loader, epoch, viz=False, with_stats=False):
    model.eval()
    accu = Accumulator()

    n_layers = np.ceil((args.n_frames // args.time_sub) / args.delta).astype(int)
    assert n_layers > 1, "n_frames must be greater than delta * time_sub"
    with torch.no_grad():
        for i, x in enumerate(tqdm(valid_loader, desc="Validation")):
            ground_truth = x['data'].to(device)
            position = x['coords'].to(device)
            space_mask = x['space_mask'][0].to(device)
            time_mask = x['time_mask'][0].to(device)
            time = x['t'].to(device)
            B, T, N, S = ground_truth.shape

            (phi_0, coords_0), inpts, phi = preprocess(ground_truth, position, space_mask, time_mask, time)

            eval_time = torch.FloatTensor([(k + 1) * args.delta * args.time_sub for k in range(n_layers)]).to(device)
            eval_time = eval_time / 20

            embeddings = model.compute_embeddings(phi_0, coords_0, n_layers)
            predictions = [model.prediction_heads(embeddings[i]).reshape(*phi_0.shape) for i in range(len(embeddings))]

            outputs = [
                model.decoder(model.query(embeddings, inpts[:, t], coords_0, eval_time)[0]) for t in
                range(T)
            ]

            outputs = torch.stack(outputs, dim=1)

            costs = {'loss': criterion(outputs, ground_truth),
                     'loss_in_t': criterion(outputs[:, time_mask], ground_truth[:, time_mask]),
                     'loss_forecasts': 0}

            for k, prediction in enumerate(predictions[:-1]):
                costs['loss_forecasts'] += criterion(prediction, phi[:, (k + 1) * args.delta])
            costs['loss_forecasts'] /= len(predictions) - 1

            if args.time_sub != 1.0:
                costs['loss_out_t'] = criterion(outputs[:, ~time_mask], ground_truth[:, ~time_mask])
                if args.space_sub != 1.0:
                    costs["loss_out_t_in_s"] = criterion(outputs[:, ~time_mask][:, :, space_mask, :],
                                                         ground_truth[:, ~time_mask][:, :, space_mask, :])
                    costs["loss_out_t_out_s"] = criterion(outputs[:, ~time_mask][:, :, ~space_mask, :],
                                                          ground_truth[:, ~time_mask][:, :, ~space_mask, :])
            if args.space_sub != 1.0:
                costs["loss_in_t_in_s"] = criterion(outputs[:, time_mask][:, :, space_mask, :],
                                                    ground_truth[:, time_mask][:, :, space_mask, :])
                costs["loss_in_t_out_s"] = criterion(outputs[:, time_mask][:, :, ~space_mask, :],
                                                     ground_truth[:, time_mask][:, :, ~space_mask, :])
            accu.add(costs, len(ground_truth))

    results = accu.mean(prefix="validation/", with_stats=with_stats)
    results['epoch'] = epoch
    print(f"=== EPOCH {epoch + 1} ===\n{results}")
    return results


def schedule(W, k=128):
    """ Return k integer values in [0, W.shape] with probability proportional to W """
    W = W.flatten()
    probability = W.cpu().numpy() / W.cpu().numpy().sum()
    k = np.random.choice(np.arange(W.shape[0]), size=k, replace=False, p=probability)
    return k


def preprocess(ground_truth, position, space_mask, time_mask, time, n_frames=args.n_frames):
    B, T, N, S = ground_truth.shape

    position = (position - position.min()) / (position.max() - position.min())

    phi_0 = ground_truth[:, 0, space_mask, :].view(B, -1, S)
    coords_0 = position[:, space_mask, :]
    phi = ground_truth[:, :, space_mask, :][:, time_mask]

    coords = position.unsqueeze(1).repeat(1, T, 1, 1).view(B, T, -1, position.shape[-1])
    time = torch.arange(n_frames).reshape(1, -1, 1).to(device)

    time = time.repeat(B, 1, coords.shape[2]).unsqueeze(-1)
    time = time / 20

    inpts = torch.cat([coords, time], dim=-1)

    return (phi_0, coords_0), inpts, phi


def main():
    print(args)

    n_layers = np.ceil((args.n_frames // args.time_sub) / args.delta).astype(int)
    assert n_layers > 1, "n_frames must be greater than delta * time_sub"
    print("Number of layers:", n_layers)

    name = args.name if args.name != '' else ""

    dataset = datasets[args.dataset]
    train_dataset = dataset(mode='train', n_frames=args.n_frames, space_subsampling=args.space_sub,
                            time_subsampling=args.time_sub)
    train_loader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=True, num_workers=NUM_WORKER,
                              pin_memory=True)

    valid_dataset = dataset(mode='valid', n_frames=args.n_frames, space_subsampling=args.space_sub,
                            time_subsampling=args.time_sub)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCHSIZE, shuffle=False, num_workers=NUM_WORKER,
                              pin_memory=True)

    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model = GNS(state_size=train_dataset.state_dim, coord_size=train_dataset.coord_dim,
                latent_size=128, gnn_density=args.gnn_density, aggregation=args.aggregation,
                activation=args.activation).to(device)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("#params:", params)

    memory = torch.inf
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, gamma=0.5, milestones=[2500, 3000, 3500, 4000])
    W = None

    for epoch in range(args.epochs):
        model.train()
        for i, x in enumerate(tqdm(train_loader, desc=f"({epoch % 20}/20)")):
            ground_truth = x['data'].to(device)
            position = x['coords'].to(device)
            space_mask = x['space_mask'][0].to(device)
            time_mask = x['time_mask'][0].to(device)
            time = x['t'].to(device)
            B, T, N, S = ground_truth.shape

            (phi_0, coords_0), inpts, phi = preprocess(ground_truth, position, space_mask, time_mask, time)
            inpts = inpts[:, time_mask][:, :, space_mask, :].view(B, -1, inpts.shape[-1])

            if W is None:
                W = torch.ones_like(inpts[0, :, 0]).to(device)

            scheduling_mask = schedule(W, k=args.n_points)
            inpts = inpts[:, scheduling_mask, :]
            target = phi.view(B, -1, S)[:, scheduling_mask, :]

            gnn_subsampling = torch.rand(phi_0.shape[1]) < args.gnn_subsampling
            phi_0 = phi_0[:, gnn_subsampling, :]
            coords_0 = coords_0[:, gnn_subsampling, :]

            eval_time = torch.FloatTensor([(k + 1) * args.delta * args.time_sub for k in range(n_layers)]).to(device)
            eval_time = eval_time / 20

            outputs, predictions = model(phi_0, coords_0, inpts, n_layers, eval_time)

            costs = {'loss_continuous': criterion(outputs, target)}

            costs['loss_forecasts'] = 0
            phi = phi[:, :, gnn_subsampling, :]
            for k, prediction in enumerate(predictions[:-1]):
                costs['loss_forecasts'] += criterion(prediction, phi[:, (k + 1) * args.delta])
            costs['loss_forecasts'] /= len(predictions) - 1
            costs['loss_total'] = costs['loss_continuous'] + args.w_forecast * costs['loss_forecasts']

            if not torch.isnan(costs['loss_total']):
                weight_mask = torch.zeros_like(W)
                weight_mask[scheduling_mask] = 1
                W = costs['loss_continuous'].detach().cpu() * weight_mask + (
                        W + costs['loss_continuous'].detach().cpu()) * (1 - weight_mask)

                optim.zero_grad()
                costs['loss_total'].backward()
                max_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1).max()
                costs['max_norm'] = max_norm
                optim.step()
            else:
                print("WARNING: NaN encountered in loss function")
            if i % 10 == 0:
                print(f"Epoch {epoch + 1} | Loss: {costs['loss_total'].item()}")

        scheduler.step()

        if epoch % 20 == 0:
            loss = validate(model, valid_loader, epoch)['validation/loss']

            if loss < memory:
                memory = loss.item()
                if args.save == "true":
                    os.makedirs(SAVE_PATH, exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(SAVE_PATH, f"{name}.pth"))


if __name__ == '__main__':
    if args.epochs == 0:
        evaluate()
    else:
        main()
