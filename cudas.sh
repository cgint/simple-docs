#!/bin/bash
sudo service ollama stop

CUDA_VISIBLE_DEVICES=0 OLLAMA_HOST=0.0.0.0:11431 ollama serve &
CUDA_VISIBLE_DEVICES=1 OLLAMA_HOST=0.0.0.0:11432 ollama serve &
CUDA_VISIBLE_DEVICES=2 OLLAMA_HOST=0.0.0.0:11433 ollama serve &
CUDA_VISIBLE_DEVICES=3 OLLAMA_HOST=0.0.0.0:11434 ollama serve