#!/bin/bash


# --------------------------------------------------------- #
# AUTHOR: Alberto M. Esmoris Pena                           #
# BRIEF: Script to install HELIOS++ in Linux Ubuntu 24      #
# --------------------------------------------------------- #


# Get script dir
SCRIPT_DIR=$(dirname -- "$(readlink -f "$0")")


# Clone repository
cd $HOME
mkdir -p git
cd git
git clone -b dev https://github.com/3dgeo-heidelberg/helios


# Install dependencies
cd helios
mkdir -p lib
scripts/build/build_all_libs.sh  -p 3.12


# Build HELIOS++
cp -r "${SCRIPT_DIR}/helios_cmake/"* .
mkdir cmake-build-release
cd cmake-build-release
cmake -DBUILD_TYPE=Release -DPYTHON_VERSION=312 -DPYTHON_BINDING=1 -DBOOST_DYNAMIC_LIBS=0 \
    -DPCL_BINDING=0 -DBUDDING_METRICS=0 -DDATA_ANALYTICS=0 \
    ..
make -j 4

# Copy HELIOS++ binary
cp helios ../helios
