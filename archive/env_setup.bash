#!/bin/bash 
conda create -n torchrl_stable python=3.7
conda activate torchrl_stable
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
pip install torchrl-nightly
pip3 install "torchrl[atari,dm_control,gym_continuous,rendering,tests,utils]"
python3 dreamer.py