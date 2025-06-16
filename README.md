This is a simultor code for QASVS and the comparison algorithms.
QASVS includes the training for GP model and DRL model.

The environment setup
Anaconda is suggested to be installed to manage the test environments.

Prerequisites
Linux or macOS
Python >=3.6
numpy, pandas
CPU or NVIDIA GPU + CUDA CuDNN
tensorflow >=1.15
Overview
The reinforcement learning method corresponding to QASVS is located in the DRL_model_training/ directory. Among them, there are many auxiliary files, such as env.py, which is the code for simulating ABR virtual playback. The main logic files for training the algorithm are located in this directory.

Baseline algorithm
PDAS,incendio,dashlet are the codes for the comparison algorithms, respectively.

Network throughput traces
Public bandwidth tracking is placed in this directory ..

Usage
When running QASVS on its own, you can execute it using the /src/driver/abr/QASVS.sh script by running

In this script, you can set various training parameters, such as specifying the directory to save the results in using --save-dir , specifying the training set using --train-trace-dir , and specifying the validation set using --val-trace-dir .

Plot a example result

cd plt
python F_1.py
