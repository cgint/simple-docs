#!/bin/bash

# sudo apt update \
#  && sudo apt upgrade -y \
#  && sudo apt install -y git \
#  && mkdir dev && cd dev && git clone https://github.com/cgint/simple-docs.git && cd simple-docs \
#  && echo "Now run either sh init_nvidia.sh or sh init_no_nvidia.sh"

sh init_no_nvidia.sh

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt upgrade -y && sudo apt install -y nvidia-container-toolkit nvidia-driver-545
sudo systemctl daemon-reload
sudo service docker restart

echo "INIT: Ollama ..."
curl -fsSL https://ollama.com/install.sh | sh
echo '[Service]"' >> environment.conf
echo 'Environment="OLLAMA_HOST=0.0.0.0:11434"' >> environment.conf
sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo mv environment.conf /etc/systemd/system/ollama.service.d/environment.conf
sudo systemctl daemon-reload
sudo service ollama restart
ollama pull neural-chat


echo "Now is the time to restart ... NVIDIA-Driver needs a restart to take effect."