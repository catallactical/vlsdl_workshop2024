# Use the official Ubuntu 22.04 base image
FROM ubuntu:24.04

# Set environment variable to prevent some interaction issues
ENV DEBIAN_FRONTEND=noninteractive

# Update package list and install basic utilities including bash
RUN apt-get upgrade && apt-get update && apt-get install -y \
  bash \
  vim \
  sudo \
  curl \
  wget \
  net-tools \
  iputils-ping \
  && rm -rf /var/lib/apt/lists/*

COPY helios_cmake /git/helios_cmake
COPY helios_linux_install.sh /git/helios_linux_install.sh
COPY ubuntu_dependencies.sh /git/ubuntu_dependencies.sh


WORKDIR /git

RUN chmod +x helios_linux_install.sh ubuntu_dependencies.sh
RUN ./ubuntu_dependencies.sh
RUN ./helios_linux_install.sh

WORKDIR /root

# Set the default shell to bash
CMD ["/bin/bash"]
