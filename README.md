# Space and time continuous physics simulation from partial observations
[![report](https://img.shields.io/badge/ArXiv-Paper-red)](https://arxiv.org/abs/2401.09198)
[![Static Badge](https://img.shields.io/badge/Project-Page-blue)](https://continuous-pde.github.io/)

![Teaser](teaser.gif)

> [**Space and Time Continuous Physics Simulation From Partial Observations**](./),            
> [Steeven Janny](https://steevenjanny.github.io/),
> [Madiha Nadri](https://madihanadri.github.io/),
> [Julie Digne](https://perso.liris.cnrs.fr/julie.digne/),
> [Christian Wolf](https://chriswolfvision.github.io/www/)       
> *International Conference on Learning Representation (ICLR), 2024*

Pytorch implementation of the paper.

## Abstract
Modern techniques for physical simulations rely on numerical schemes and mesh-refinement methods to address trade-offs between precision and complexity, but these handcrafted solutions are tedious and require high computational power. Data-driven methods based on large-scale machine learning promise high adaptivity by integrating long-range dependencies more directly and efficiently. In this work, we focus on fluid dynamics and address the shortcomings of a large part of the literature, which are based on fixed support for computations and predictions in the form of regular or irregular grids. We propose a novel setup to perform predictions in a continuous spatial and temporal domain while being trained on sparse observations. We formulate the task as a double observation problem and propose a solution with two interlinked dynamical systems defined on, respectively, the sparse positions and the continuous domain, which allows to forecast and interpolate a solution from the initial condition. Our practical implementation involves recurrent GNNs and a spatio-temporal attention observer capable of interpolating the solution at arbitrary locations. Our model not only generalizes to new initial conditions (as standard auto-regressive models do) but also performs evaluation at arbitrary space and time locations. We evaluate on three standard datasets in fluid dynamics and compare to strong baselines, which are outperformed both in classical settings and in the extended new task requiring continuous predictions.

## Generate data
For **Navier-Stokes** dataset, you can generate the data by running the following command:
```bash 
python DatasetGeneration/gen_NavierStokes.py --mode [train, valid or test]
```
(Note that the data generation script is based on the one provided by [Dino](https://github.com/mkirchmeyer/DINo).

For **Shallow Water** dataset, use the provided link from [DINo](https://github.com/mkirchmeyer/DINo) and 
apply the post-processing script to generate the dataset in the same format as the Navier-Stokes dataset:
```bash
python DatasetGeneration/process_ShallowWater.py
```

Finally, for **Eagle** dataset, you can download the data from the [official website](https://eagle-dataset.github.io/).


## Training

To train the model, you can use the following command:
```bash
python train.py --epochs 4500 \
  --lr 0.001 \
  --n_frames 21 \
  --dataset "navier" \
  --space_sub 0.10 \
  --time_sub 2 \
  --gnn_density 4 \
  --delta 2 \
  --name "navier-10-2"
```

## Evaluation
Simply set the number of epochs to 0. The model will be loaded from the latest checkpoint and evaluated on the test set.
```bash
python train.py --epochs 0 \
  --n_frames 21 \
  --dataset "navier" \
  --space_sub 0.10 \
  --time_sub 2 \
  --gnn_density 4 \
  --delta 2 \
  --name "navier-10-2"
```

## Citation

If you find our work useful please cite our paper:

```
@inproceedings{janny2024space,
  title={Space and Time Continuous Physics Simulation From Partial Observations},
  author={Janny, Steeven and Nadri, Madiha and Digne, Julie and Wolf, Christian},
  booktitle={International Conference on Learning Representation (ICLR)},
  year={2024}
}
```

## License

Code is distributed under the CC BY-NC-SA 4.0 License. See [LICENSE](LICENSE) for more information.