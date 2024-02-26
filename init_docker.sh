#!/bin/bash

# sudo apt update \
#  && sudo apt upgrade -y \
#  && sudo apt install -y git \
#  && mkdir dev && cd dev && git clone https://github.com/cgint/simple-docs.git && cd simple-docs \
#  && echo "Now run either sh init_nvidia.sh or sh init_no_nvidia.sh"

sudo apt install -y curl wget atop docker-compose docker-buildx net-tools python3.11 python3.11-venv python3-pip
sudo apt autoremove -y && sudo apt clean all
sudo usermod -a -G docker christian.gintenreiter
sudo systemctl enable docker
sudo service docker start