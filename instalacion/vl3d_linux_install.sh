#!/bin/bash

# Install Ubuntu docker dependencies
apt update
apt install -y wget sudo mesa-common-dev libglu1-mesa-dev libusb-1.0-0-dev 

# Install miniconda
mkdir -p $HOME/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $HOME/miniconda3/miniconda.sh
bash $HOME/miniconda3/miniconda.sh -b -u -p $HOME/miniconda3
rm $HOME/miniconda3/miniconda.sh

# Add miniconda to PATH
export PATH=$PATH:$HOME/miniconda3/bin/

# Configure conda
source $HOME/miniconda3/etc/profile.d/conda.sh

# Create VL3D environment
cd vl3d
conda env create -f vl3d_lin.yml
conda activate vl3d

# Install C++ bindings
mkdir -p cpp/lib && cd cpp/lib
./ubuntu_deps.sh
./lib_install.sh

# Compile C++ bindings
cd ..
mkdir -p build
cd build
cmake -DBUILD_TYPE=Release -DPython3_FIND_VIRTUALENV=ONLY ..
make -j 8
cd ..

# Execute tests
cd ..
python vl3d.py --test