#!/bin/bash

conda activate Bandit

python semi_synthetic_exp.py -T 10000000 -o ./Results -t 5 -z ./Dataset/Yahoo-Front-Page/R6 -s 1000 -e 1000
python yahoo_plot.py

conda deactivate