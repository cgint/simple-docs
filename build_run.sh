#!/bin/bash

# Define a variable for the image name
IMAGE_NAME="llama_simple_docs"

mkdir -p data/fastembed_cache

# Build the Docker image and run if successful
docker build -t $IMAGE_NAME . && time docker run -it --rm \
    --name "simple_docs" \
    -v "./data:/data" \
    -v "./data_root_nltk:/root/nltk_data" \
    -v "./data/root_cache_huggingface:/root/.cache/huggingface" \
    -e TOGETHER_AI_KEY \
    -e "PARAM_COMMAND=$1" \
    -e "LLM_ENGINE=ollama" \
    -e "LLM_MODEL=neural-chat" \
    -e "ASK_SENTENCES=Was sind die öffnungszeiten von poscher ?" \
    $IMAGE_NAME
    # -e "ASK_SENTENCES=Tell me about alerting in distributed systems###What is AIOPs ?###What are the core challenges of alerting ?" \
    # -e "LLM_ENGINE=together" \
    # -e "LLM_MODEL=mixtral-together" \
    # -e "OPENAI_MODEL=mistralai/Mixtral-8x7B-Instruct-v0.1" \
    
# sudo chown `whoami` data/*