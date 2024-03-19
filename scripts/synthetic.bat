@echo off

conda activate Bandit

python all_simulations.py -o ./All_Sim_Final -n Final
python plot_regret.py -o ./All_Sim_Final -n Final

conda deactivate