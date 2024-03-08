#!/bin/bash
HOST_IP_ADDR=$(ifconfig | egrep "inet (192\.|10\.)" | head -n 1 | awk '{print $2}')
echo "HOST_IP_ADDR: $HOST_IP_ADDR"
PLAYGROUND="pdf-2-chat"
echo "PLAYGROUND: $PLAYGROUND"



# Only execute 'docker-compose up -d' when /$PLAYGROUND/ can be found in docker-compose.yml
if grep -q $PLAYGROUND docker-compose.yaml; then
  echo "docker-compose up -d"
  docker-compose up -d
else
  echo "$PLAYGROUND not configured in docker-compose.yml. Exiting."
  exit 1
fi

#  LLM_ENGINE=together LLM_MODEL=mixtral-together OPENAI_MODEL=mistralai/Mixtral-8x7B-Instruct-v0.1 \
#  LLM_ENGINE=ollama-ssh LLM_MODEL=neural-chat HOST_IP_OLLAMA=localhost \
#  LLM_ENGINE=ollama-multi HOST_IP_OLLAMA=192.168.0.99 LLM_MODEL=neural-chat \
#  LLM_ENGINE=ollama HOST_IP_OLLAMA=localhost LLM_MODEL=neural-chat \
#  LLM_ENGINE=groq LLM_MODEL=mixtral-8x7b-32768 \
#  LLM_ENGINE=ollama-multi-local-4 LLM_MODEL=neural-chat \
# QUERY_ENGINE_VARIANT=vector-graph-term-kggraph-docsum-bm25

PARAM_COMMAND=ask QUERY_ENGINE_VARIANT=vector-graph-term-kggraph-docsum-bm25 HOST_IP=$HOST_IP_ADDR \
 DATA_PLAYGROUND=$PLAYGROUND \
 LLM_ENGINE=together LLM_MODEL=mistral-together OPENAI_MODEL=mistralai/Mistral-7B-Instruct-v0.2 \
 RERANKER_MODEL=BAAI/bge-reranker-base RERANKER_K=10 \
 DATA_DIR=./data \
 python3.11 simple_docs.py

