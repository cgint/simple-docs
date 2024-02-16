#!/bin/bash
docker-compose up -d

HOST_IP_ADDR=$(ifconfig | grep "192\." | head -n 1 | awk '{print $2}')
echo "HOST_IP_ADDR: $HOST_IP_ADDR"

#  LLM_ENGINE=together LLM_MODEL=mistral-together OPENAI_MODEL=mistralai/Mistral-7B-Instruct-v0.2 \
#  LLM_ENGINE=together LLM_MODEL=mixtral-together OPENAI_MODEL=mistralai/Mixtral-8x7B-Instruct-v0.1 \
#  DEL_INDICES_ALL=true \
#  LLM_ENGINE=ollama LLM_MODEL=neural-chat \
#  LLM_ENGINE=ollama-ssh LLM_MODEL=neural-chat HOST_IP_OLLAMA=localhost \

time PARAM_COMMAND=index INDEXING_ENGINE_VARIANT==vector-graph-term-kggraph HOST_IP=$HOST_IP_ADDR \
 DATA_PLAYGROUND=git-whoop \
 RERANKER_MODEL=BAAI/bge-reranker-base RERANKER_K=5 \
 DATA_DIR=./data \
 LLM_ENGINE=ollama-multi HOST_IP_OLLAMA=192.168.0.99 LLM_MODEL=neural-chat \
 python3.11 simple_docs.py

