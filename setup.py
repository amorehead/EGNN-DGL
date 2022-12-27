#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name='EGNN-DGL',
    version='1.0.0',
    description='An implementation of the Equivariant Graph Neural Network (EGNN) layer type for DGL-PyTorch.',
    author='Alex Morehead',
    author_email='alex.morehead@gmail.com',
    url='https://github.com/amorehead/EGNN-DGL',
    install_requires=[
        'setuptools==65.5.1',
        'torchmetrics==0.6.2',
        'wandb==0.12.9',
        'pytorch-lightning==1.6.0',
        'fairscale==0.4.4'
    ],
    packages=find_packages(),
)
