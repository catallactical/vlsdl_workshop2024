#!/bin/bash

# Get script dir
SCRIPT_DIR=$(dirname -- "$(readlink -f "$0")")

IMG_NAME=helios/workshop:latest
CONTAINER_NAME=helios_workshop

docker build -t $IMG_NAME .
docker run -v $SCRIPT_DIR/..:/root/vlsdl_workshop2024 -it --name $CONTAINER_NAME --memory=8g $IMG_NAME /bin/bash 
