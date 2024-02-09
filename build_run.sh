#!/bin/bash

# Define a variable for the image name
IMAGE_NAME="llama_simple_docs"

mkdir -p data/fastembed_cache
mkdir -p data/index_inbox
mkdir -p data/index_inbox/done
mkdir -p data/html_dl_cache
mkdir -p data/root_cache_huggingface
mkdir -p data/term_data
mkdir -p data/aim

loop_through() {
    local -a QUERY_ENGINE_VARIANTS=($2)  # Convert first argument back to array
    for QUERY_ENGINE_VARIANT in "${QUERY_ENGINE_VARIANTS[@]}"; do
        # Backup aim-files to have a history just in case
        CUR_TIME_FOR_FILE=$(date +"%Y%m%d_%H%M%S")
        echo "Creating backup of aim-files with timestamp: $CUR_TIME_FOR_FILE"
        (cd data/; tar -czf aim-backup_$CUR_TIME_FOR_FILE.tgz aim)

        echo "QUERY_ENGINE_VARIANT: $QUERY_ENGINE_VARIANT"
        echo "ASK_SENTENCE: $3"
        docker run -it --rm \
            --name "simple_docs" \
            -v "./data:/data" \
            -v "./data_root_nltk:/root/nltk_data" \
            -v "./data/root_cache_huggingface:/root/.cache/huggingface" \
            -e "HOST_IP=$4" \
            -e TOGETHER_AI_KEY \
            -e "PARAM_COMMAND=$1" \
            -e "QUERY_ENGINE_VARIANT=$QUERY_ENGINE_VARIANT" \
            -e "LLM_ENGINE=together" \
            -e "LLM_MODEL=mixtral-together" \
            -e "OPENAI_MODEL=mistralai/Mixtral-8x7B-Instruct-v0.1" \
            -e "ASK_SENTENCES=$3" \
            $IMAGE_NAME
            # -e "LLM_ENGINE=ollama" \
            # -e "LLM_MODEL=neural-chat" \
    # -e "ASK_SENTENCES=OllamaEmbeddingsTell me about alerting in distributed systems###What is AIOPs ?###What are the core challenges of alerting ?" \
    # -e "ASK_SENTENCES=Was sind die öffnungszeiten von poscher ?" \
    # --gpus all \
    done
}

HOST_IP_ADDR=$(ifconfig | grep 192 | head -n 1 | awk '{print $2}')
echo "HOST_IP_ADDR: $HOST_IP_ADDR"

# Convert arrays to strings before passing to function
ENGINE_VARIANTS=("vector-graph-docsum-bm25" "vector-docsum-bm25" "vector-graph-bm25" "vector-graph" "vector-docsum" "vector-bm25" "vector-graph-docsum" "graph-docsum" "graph-bm25" "docsum-bm25")
# ENGINE_VARIANTS=("vector-graph-docsum-bm25" "vector" "graph" "docsum" "bm25")
docker build -t $IMAGE_NAME . \
  && loop_through "$1" "${ENGINE_VARIANTS[*]}" "Can i use Ollama Embeddings in llama index ?" "$HOST_IP_ADDR" \
  && loop_through "$1" "${ENGINE_VARIANTS[*]}" "How specifically can i use Ollama Embeddings in llama index, which implementations are there ?" "$HOST_IP_ADDR"
    
# sudo chown -R `whoami` data/