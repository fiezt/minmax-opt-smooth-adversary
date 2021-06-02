# Overview

This repository contains the code for experiments in the paper: 

**Minimax Optimization with Smooth Algorithmic Adversaries**

by Tanner Fiez, Chi Jin, Praneeth Netrapalli, and Lillian Ratliff.

For any questions on this repository, feel free to contact me by email at fiezt@uw.edu.


## Dependencies
The conda environment used for the experiments has been provided in environment.yml

## Contents

The code is split between directories: 
* The DiracGan directory has the code for the DiracGAN experiments.
* The MoG_Gan directory has the code for the Mixture of Gaussian GAN experiments. 
* The adversarial_training directory has the code for the adversarial training experiments.

#### DiracGan Directory

This folder has a standalone jupyter notebook for the experiment.

#### MOG_Gan Directory

The primary code for the experiments is contained in exps.py. \
The model, data generation code, and plotting utilities are contained in model.py, data.py, and utils.py respectively.

To configs folder contains configuration files that were used for the experiments in the paper. \
These experiments can be run using the basic calls contained in run_exps.sh.


#### adversarial_training Directory

The primary code for the training is in experiments.py.\
The attack code is in adversary.py and the data proceesing code is in dataprocess.py 

To config directory contains configuration files for each algorithm considered in the experiments.\
The experiments can be run with run_sims.sh. This will save the models and some of the attack results.\
To get the full attack results, after running run_sims.sh, you will also need to run budget_attack.py and budget_adam_attack.py. \
Finally, the gradient norm results can be obtained calling python compute_grad_norms.py.

The results can then be processed and plotted using the AnalyzeResults.ipynb notebook.


## Acknowledgements
The code for the mixture of Gaussian experiments has been adapted\
from an implementation for unrolled gans available at https://github.com/MarisaKirisame/unroll_gan. 

The code for the adversarial training has been adapted from https://github.com/optimization-for-data-driven-science/Robust-NN-Training.
