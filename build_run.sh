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
        # CUR_TIME_FOR_FILE=$(date +"%Y%m%d_%H%M%S")
        # echo "Creating backup of aim-files with timestamp: $CUR_TIME_FOR_FILE"
        # (cd data/; tar -I pigz -cf aim-backup_$CUR_TIME_FOR_FILE.tgz aim)

        echo "QUERY_ENGINE_VARIANT: $QUERY_ENGINE_VARIANT"
        echo "ASK_SENTENCE: $3"
            # --runtime nvidia --gpus all \
        docker run -it --rm \
            --name "simple_docs" \
            --privileged \
            -v "./data:/data" \
            -v "./data_root_nltk:/root/nltk_data" \
            -v "./data/root_cache_huggingface:/root/.cache/huggingface" \
            -e "HOST_IP=$4" \
            -e TOGETHER_AI_KEY \
            -e OPENAI_API_KEY \
            -e "PARAM_COMMAND=$1" \
            -e "QUERY_ENGINE_VARIANT=$QUERY_ENGINE_VARIANT" \
            -e "RERANKER_MODEL=BAAI/bge-reranker-base" \
            -e "RERANKER_K=5" \
            -e "LLM_ENGINE=ollama" \
            -e "LLM_MODEL=neural-chat" \
            $IMAGE_NAME
            # -e "HOST_IP_OLLAMA=192.168.0.99" \
            # -e "LLM_ENGINE=together" \
            # -e "LLM_MODEL=mistral-together" \
            # -e "OPENAI_MODEL=mistralai/Mistral-7B-Instruct-v0.2" \
            # -e "ASK_SENTENCES=$3" \
            # -e "OPENAI_MODEL=mistralai/Mixtral-8x7B-Instruct-v0.1" \
    # -e "ASK_SENTENCES=OllamaEmbeddingsTell me about alerting in distributed systems###What is AIOPs ?###What are the core challenges of alerting ?" \
    # -e "ASK_SENTENCES=Was sind die öffnungszeiten von poscher ?" \
    done
}

HOST_IP_ADDR=$(ifconfig | grep "inet 1.*\." | head -n 1 | awk '{print $2}')
echo "HOST_IP_ADDR: $HOST_IP_ADDR"

# Convert arrays to strings before passing to function
#ENGINE_VARIANTS=("bm25") # "docsum" "bm25" "vector-gaph"  # "vector-docsum-bm25" "vector-graph-bm25" "vector-graph" "vector-docsum" "vector-bm25" "vector-graph-docsum" "graph-docsum" "graph-bm25" "docsum-bm25")
ENGINE_VARIANTS=("vector-docsum-bm25")
BUILD_COMMAND="docker build"
# If a script called 'dbuild.sh' is available for execution in path not necessarily in local directory, use it instead of 'docker build'
# think about the use of 'which dbuild.sh' just to find out
which dbuild.sh > /dev/null
if [ $? -eq 0 ]; then
  BUILD_COMMAND="dbuild.sh"
fi

echo "Using BUILD_COMMAND: $BUILD_COMMAND"
$BUILD_COMMAND -t $IMAGE_NAME .
if [ $? -eq 0 ]; then
    RUN_COMMAND="ask"
    if [ "$1" == "index" ]; then
        RUN_COMMAND="$1"
    fi
    echo "Build OK - running given command '$RUN_COMMAND'"
    if [ "$RUN_COMMAND" == "index" ]; then
        loop_through "$RUN_COMMAND" "index-all" "" "$HOST_IP_ADDR"
    else
        #loop_through "$RUN_COMMAND" "${ENGINE_VARIANTS[*]}" "Can i use Ollama Embeddings in llama index ?" "$HOST_IP_ADDR"
        #loop_through "$RUN_COMMAND" "${ENGINE_VARIANTS[*]}" "How specifically can i use Ollama Embeddings in llama index, which implementations are there ?" "$HOST_IP_ADDR"
        loop_through "$RUN_COMMAND" "${ENGINE_VARIANTS[*]}" "Which properties do i have to set on a Google Ads Campaign that I manage via Google Ads API to link it to the Merchant Center ?" "$HOST_IP_ADDR"
        #loop_through "$RUN_COMMAND" "${ENGINE_VARIANTS[*]}" "Is there a way to query for brand or category when downloading products via Google Shopping API ?" "$HOST_IP_ADDR"
        #loop_through "$RUN_COMMAND" "${ENGINE_VARIANTS[*]}" "How is Meta Marketing API different than Google Ads API and what are the high level similarities ?" "$HOST_IP_ADDR"
    fi
else
    echo "Build failed"
    exit 1
fi

sudo chown -R `whoami` data/