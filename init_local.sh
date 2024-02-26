#!/bin/bash

# sudo apt update \
#  && sudo apt upgrade -y \
#  && sudo apt install -y git curl wget atop python3.11 python3.11-venv python3-pip \
#  && mkdir dev && cd dev && git clone https://github.com/cgint/simple-docs.git && cd simple-docs \
#  && sh init_local.sh

sh init_venv.sh &
sh init_ollama.sh &
sh init_docker.sh
sh init_nvidia.sh