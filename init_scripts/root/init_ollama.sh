#!/bin/bash

echo "INIT: Ollama ..."
curl -fsSL https://ollama.com/install.sh | sh
echo '[Service]' >> environment.conf
echo 'Environment="OLLAMA_HOST=0.0.0.0:11434"' >> environment.conf
mkdir -p /etc/systemd/system/ollama.service.d
mv environment.conf /etc/systemd/system/ollama.service.d/environment.conf
systemctl daemon-reload
service ollama restart
ollama pull neural-chat