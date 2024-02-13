import os

host_ip = os.getenv("HOST_IP", "host.docker.internal")
host_ip_ollama = os.getenv("HOST_IP_OLLAMA", host_ip)
data_base_dir = os.getenv("DATA_DIR", "/data") # set to "./data" when running outside docker
guidance_gpt_version = os.getenv("GUIDANCE_GPT_VERSION", "gpt-4-0125-preview") # "gpt-3.5-turbo"

max_files_to_index_per_run = int(os.getenv("MAX_FILES_INDEX_PER_RUN", 1000)) # None for no limit
index_dir = data_base_dir + "/index_inbox"
index_dir_done = data_base_dir + "/index_inbox_done"
term_data_dir = data_base_dir + "/term_data"
vector_storage_dir = data_base_dir + "/vector_index"
collection = vector_storage_dir.replace("/", "_").replace("_", "").replace(".", "")
doc_sum_index_dir = data_base_dir + "/doc_sum_index"
aim_dir = data_base_dir + "/aim"
graph_db = "neo4j"
embed_cache_dir = data_base_dir + "/fastembed_cache/"
