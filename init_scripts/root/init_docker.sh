#!/bin/bash

# apt update \
#  && apt upgrade -y \
#  && apt install -y git curl wget atop python3.11 python3.11-venv python3-pip \
#  && mkdir dev && cd dev && git clone https://github.com/cgint/simple-docs.git && cd simple-docs \
#  && echo "Now run either sh init_nvidia.sh or sh init_no_nvidia.sh"

apt install -y docker-compose docker-buildx net-tools
apt autoremove -y && apt clean all
usermod -a -G docker christian.gintenreiter
systemctl enable docker
service docker start
