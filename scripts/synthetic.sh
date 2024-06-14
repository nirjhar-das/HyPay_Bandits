#!/bin/bash

conda activate Bandit

python all_simulations.py -o ./Results -n Final
python plot_regret.py -o ./Results -n Final

conda deactivate