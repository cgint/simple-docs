#!/bin/bash

# apt update \
#  && apt upgrade -y \
#  && apt install -y git curl wget atop python3.11 python3.11-venv python3-pip \
#  && mkdir dev && cd dev && git clone https://github.com/cgint/simple-docs.git && cd simple-docs \
#  && sh init_scripts/root/init_remote.sh

sh init_scripts/root/init_venv.sh &
sh init_scripts/root/init_docker.sh
