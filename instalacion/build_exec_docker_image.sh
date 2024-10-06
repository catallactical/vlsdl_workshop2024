#!/bin/bash

# Get script dir
SCRIPT_DIR=$(dirname -- "$(readlink -f "$0")")

IMG_NAME=helios/workshop:latest
CONTAINER_NAME=helios_workshop

docker build -t $IMG_NAME .
docker run -v $SCRIPT_DIR/..:/root/vlsdl_workshop2024 -p 8888:8888 -it --name $CONTAINER_NAME $IMG_NAME /bin/bash 
