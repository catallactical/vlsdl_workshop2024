#!/bin/bash


# --------------------------------------------------------- #
# AUTHOR: Alberto M. Esmoris Pena                           #
# BRIEF: Script to install HELIOS++ in Linux Ubuntu 24      #
# --------------------------------------------------------- #


# Get script dir
SCRIPT_DIR=$(dirname -- "$(readlink -f "$0")")

PYTHON_DOT_VERSION=3.12

# Remove dot from version
PYTHON_VERSION=$(echo $PYTHON_DOT_VERSION | tr -d '.')

# Clone repository
cd $HOME
mkdir -p git
cd git
git clone -b devel https://github.com/albertoesmp/helios.git


# Change paths in helios/local_env.sh with sed
sed -i 's/HELIOS_DIR=.*/HELIOS_DIR=$HOME\/git\/helios\//' $HOME/git/helios/scripts/local_env.sh

# Install dependencies
cd helios
mkdir -p lib
scripts/build/build_all_libs.sh -p $PYTHON_DOT_VERSION


# Build HELIOS++
cp -r "${SCRIPT_DIR}/helios_cmake/"* .
mkdir cmake-build-release
cd cmake-build-release
cmake -DBUILD_TYPE=Release -DPYTHON_VERSION=$PYTHON_VERSION -DPYTHON_BINDING=1 -DBOOST_DYNAMIC_LIBS=0 -DPCL_BINDING=0 -DBUDDING_METRICS=0 -DDATA_ANALYTICS=0 ..
make -j 4

# Copy HELIOS++ binary
cp helios ../helios
