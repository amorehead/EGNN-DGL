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
        'setuptools==57.4.0',
        'torchmetrics==0.5.1',
        'wandb==0.12.2',
        'pytorch-lightning==1.4.8',
        'fairscale==0.4.0'
    ],
    packages=find_packages(),
)
