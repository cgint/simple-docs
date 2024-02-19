#!/bin/bash

# sudo apt update \
#  && sudo apt upgrade -y \
#  && sudo apt install -y git \
#  && mkdir dev && cd dev && git clone https://github.com/cgint/simple-docs.git && cd simple-docs \
#  && sh init.sh

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt install -y curl wget atop docker-compose docker-buildx net-tools python3.11 python3.11-venv python3-pip nvidia-container-toolkit nvidia-driver-545 cuda-drivers-545
sudo usermod -a -G docker christian.gintenreiter
sudo systemctl enable docker
sudo apt autoremove

 
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "INIT: Ollama ..."
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl disable ollama # will be started manually
ollama pull neural-chat

echo "To start ollama hit: OLLAMA_HOST=0.0.0.0:11434 ollama serve"