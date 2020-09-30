#!/bin/bash
# nvim 	:PlugInstall && :set background=dark #for gruvbox

# Must run script as source ec2setup2.sh in order to source pytorch_latest_p36 below
source activate pytorch_latest_p36
# download project
git clone https://github.com/phelps-matthew/FeatherMap.git
cd ./FeatherMap/
pip install -e .
cd feathermap
# test to ensure proper operation
python ffnn_main.py --epochs 1
