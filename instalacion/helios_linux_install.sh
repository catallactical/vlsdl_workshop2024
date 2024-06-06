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
git clone -b devel-aux https://github.com/3dgeo-heidelberg/helios


# Install dependencies
cd helios
mkdir -p lib
#scripts/build/build_all_libs.sh  -p 3.12  # TODO Remove


# Build HELIOS++
cp -r "${SCRIPT_DIR}/helios_cmake/"* .
mkdir cmake-build-release
cd cmake-build-release
# TODO Remoove echos below
echo '---------------------------------------------------------'
echo 'PWD: '$PWD
echo '---------------------------------------------------------'
cmake -DBUILD_TYPE=Release -DPYTHON_VERSION=312 -DPYTHON_BINDING=1 -DBOOST_DYNAMIC_LIBS=0 \
    -DPCL_BINDING=0 -DBUDDING_METRICS=0 -DDATA_ANALYTICS=0 \
    ..
make -j 8
