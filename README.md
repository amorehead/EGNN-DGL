# EGNN-DGL

An implementation of the Equivariant Graph Neural Network (EGNN) layer type for DGL-PyTorch.

## Citing this work

If you use this implementation of the EGNN, please cite the original authors:

```bibtex
@InProceedings{pmlr-v139-satorras21a,
  title = 	 {E(n) Equivariant Graph Neural Networks},
  author =       {Satorras, V\'{\i}ctor Garcia and Hoogeboom, Emiel and Welling, Max},
  booktitle = 	 {Proceedings of the 38th International Conference on Machine Learning},
  pages = 	 {9323--9332},
  year = 	 {2021},
  editor = 	 {Meila, Marina and Zhang, Tong},
  volume = 	 {139},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {18--24 Jul},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v139/satorras21a/satorras21a.pdf},
  url = 	 {https://proceedings.mlr.press/v139/satorras21a.html},
  abstract = 	 {This paper introduces a new model to learn graph neural networks equivariant to rotations, translations, reflections and permutations called E(n)-Equivariant Graph Neural Networks (EGNNs). In contrast with existing methods, our work does not require computationally expensive higher-order representations in intermediate layers while it still achieves competitive or better performance. In addition, whereas existing methods are limited to equivariance on 3 dimensional spaces, our model is easily scaled to higher-dimensional spaces. We demonstrate the effectiveness of our method on dynamical systems modelling, representation learning in graph autoencoders and predicting molecular properties.}
}
```

## Installation

First, install and configure Conda environment:

```bash
# Clone this repository:
git clone https://github.com/amorehead/EGNN-DGL

# Change to project directory:
cd EGNN-DGL
EGNN_DGL=$(pwd)

# Set up Conda environment locally
conda env create --name EGNN-DGL -f environment.yml

# Activate Conda environment located in the current directory:
conda activate EGNN-DGL

# (Optional) Perform a full install of the pip dependencies described in 'requirements.txt':
pip3 install -r requirements.txt

# (Optional) To remove the long Conda environment prefix in your shell prompt, modify the env_prompt setting in your .condarc file with:
conda config --set env_prompt '({name})'
 ```

## QM9 Dataset

Download a preprocessed version of the dataset [here](https://drive.google.com/file/d/1EpJG0Bo2RPK30bMKK6IUdsR5r0pTBEP0/view?usp=sharing) and place it in `project/datasets/QM9/`.

## Training

Hint: Run `python3 lit_model_train.py --help` to see all available CLI arguments

 ```bash
cd project
python3 lit_model_train.py --lr 1e-3 --weight_decay 1e-2
cd ..
```