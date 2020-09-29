#!/bin/bash
# scp ~/Insight/project/FeatherMap/ec2setup.sh ubuntu@ec2..:~
source activate pytorch_latest_p36
sudo apt update -y
sudo apt install neovim -y
# install dotfiles
git clone --separate-git-dir=$HOME/dotfiles https://github.com/phelps-matthew/dotfiles.git tmpdotfiles
rsync --recursive --verbose --exclude '.git' tmpdotfiles/ $HOME/
rm -r tmpdotfiles
# install vimplug
curl -fLo ~/.local/share/nvim/site/autoload/plug.vim --create-dirs https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim
# tmux plugin manager
git clone https://github.com/tmux-plugins/tpm ~/.tmux/plugins/tpm
tmux source ~/.tmux.conf

# start a new window
tmux new-session -t aws-tmx

# nvim 	:PlugInstall && :set background=dark #for gruvbox
# download project
#git clone https://github.com/phelps-matthew/FeatherMap.git
#cd ./FeatherMap/
#pip install -e .
