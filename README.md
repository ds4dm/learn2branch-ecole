# Exact Combinatorial Optimization with Graph Convolutional Neural Networks (Ecole+Pytorch+Pytorch Geometric reimplementation)

This is the official reimplementation of the proposed GNN model from the paper "Exact Combinatorial Optimization with Graph Convolutional Neural Networks" [NeurIPS 2019 paper](https://arxiv.org/abs/1906.01629) using the [Ecole library](https://github.com/ds4dm/ecole). This reimplementation also makes use [Pytorch](https://github.com/pytorch/pytorch) instead of Tensorflow, and of [Pytorch Geometric](https://github.com/pyg-team/pytorch_geometric) for handling the GNN. As a consequence, much of the code is now simplified. Slight discrepancies in results from the original implementation is to be expected.

As mentionned, this repo only implements the GNN model. For comparisons with the other ML competitors (ExtraTrees, LambdaMART and SVMRank), please see the original implementation [here](https://github.com/ds4dm/learn2branch).

<table style='border:none;'>
  <tr>
    <th><img src="https://www.ecole.ai/images/ecole-logo.png" height="120"></th>
    <th><img src="https://github.com/pytorch/pytorch/blob/master/docs/source/_static/img/pytorch-logo-dark.png" height="60"></th>
    <th><img src="https://raw.githubusercontent.com/pyg-team/pytorch_geometric/master/docs/source/_static/img/pyg1.svg?sanitize=true" height="60"></th>
  </tr>
</table>

## Authors

Maxime Gasse, Didier Chételat, Nicola Ferroni, Laurent Charlin and Andrea Lodi.

## Installation

Our recommended installation uses the [Conda package manager](https://docs.conda.io/en/latest/miniconda.html). The previous implementation required you to compile a patched version of SCIP and PySCIPOpt using Cython. This is not required anymore, as Conda packages are now available, which are dependencies of the Ecole conda package itself.

__Instructions:__ Install Ecole, Pytorch and Pytorch Geometric using conda. At the time of writing these installation instructions, this can be accomplished by running:

```
conda install ecole
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install pyg -c pyg -c conda-forge
```

Please refer to the most up to date installation instructions for [Ecole](https://github.com/ds4dm/ecole#installation), [Pytorch](https://pytorch.org/get-started/locally) and [Pytorch Geometric](https://github.com/pyg-team/pytorch_geometric#installation) if you encounter any errors.

## Running

### Set Covering
```
# Generate MILP instances
python 01_generate_instances.py setcover
# Generate supervised learning datasets
python 02_generate_samples.py setcover -j 4  # number of available CPUs
# Training
for i in {0..4}
do
    python 03_train_gnn.py setcover -s $i
done
# Evaluation
python 04_evaluate.py setcover
```

### Combinatorial Auction
```
# Generate MILP instances
python 01_generate_instances.py cauctions
# Generate supervised learning datasets
python 02_generate_samples.py cauctions -j 4  # number of available CPUs
# Training
for i in {0..4}
do
    python 03_train_gnn.py cauctions -s $i
done
# Evaluation
python 04_evaluate.py cauctions
```

### Capacitated Facility Location
```
# Generate MILP instances
python 01_generate_instances.py facilities
# Generate supervised learning datasets
python 02_generate_samples.py facilities -j 4  # number of available CPUs
# Training
for i in {0..4}
do
    python 03_train_gnn.py facilities -s $i
done
# Evaluation
python 04_evaluate.py facilities
```

### Maximum Independent Set
```
# Generate MILP instances
python 01_generate_instances.py indset
# Generate supervised learning datasets
python 02_generate_samples.py indset -j 4  # number of available CPUs
# Training
for i in {0..4}
do
    python 03_train_gnn.py indset -s $i
done
# Evaluation
python 04_evaluate.py indset
```

## Citation
Please cite our paper if you use this code in your work.
```
@inproceedings{conf/nips/GasseCFCL19,
  title={Exact Combinatorial Optimization with Graph Convolutional Neural Networks},
  author={Gasse, Maxime and Chételat, Didier and Ferroni, Nicola and Charlin, Laurent and Lodi, Andrea},
  booktitle={Advances in Neural Information Processing Systems 32},
  year={2019}
}
```

## Questions / Bugs
Please feel free to submit a Github issue if you have any questions or find any bugs. We do not guarantee any support, but will do our best if we can help.
