#!/bin/bash


# --------------------------------------------------------- #
# AUTHOR: Alberto M. Esmoris Pena                           #
# BRIEF: Script to install the software dependencies.       #
# --------------------------------------------------------- #


# Get dependencies from aptitude
apt-get install -y build-essential dkms gcc-10 g++-10 pkgconf cmake \
    sqlite3 libsqlite3-dev libtiff-dev libcurlpp-dev python3-dev
# Symbolic link to use gcc-10 and g++10
rm -f /usr/bin/gcc
ln -s /usr/bin/gcc-10 /usr/bin/gcc
rm -f /usr/bin/g++
ln -s /usr/bin/g++-10 /usr/bin/g++



