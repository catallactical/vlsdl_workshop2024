#!/bin/bash
    
apt update
apt install curl software-properties-common -y
add-apt-repository ppa:deadsnakes/ppa -y
apt update
   
apt install python3.10-dev -y
    
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
    
apt install python3.10-distutils -y
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3.10 get-pip.py
      
# Install PyHelios dependencies
pip install numpy open3d notebook
