import os
from lib.index.helper import cur_simple_date_time_sec

run_start_time_id = cur_simple_date_time_sec()

host_ip = os.getenv("HOST_IP", "host.docker.internal")
host_ip_ollama = os.getenv("HOST_IP_OLLAMA", host_ip)
guidance_gpt_version = os.getenv("GUIDANCE_GPT_VERSION", "gpt-4-0125-preview") # "gpt-3.5-turbo"
del_indices_all = os.getenv("DEL_INDICES_ALL", False)
max_files_to_index_per_run = int(os.getenv("MAX_FILES_INDEX_PER_RUN", 2000000)) # None for no limit
graph_db = "neo4j"

data_base_dir = os.getenv("DATA_DIR", "/data") # set to "./data" when running outside docker
embed_cache_dir = data_base_dir + "/fastembed_cache/"
html_dl_cache_dir = data_base_dir + "/html_dl_cache"
ignore_html_dl_cache = os.environ.get("IGNORE_HTML_DL_CACHE", "false").lower() == "true"
data_playground = os.getenv("DATA_PLAYGROUND", "playground")
data_dir = data_base_dir + "/" + data_playground
index_dir = data_dir + "/index_inbox"
index_dir_done = data_dir + "/index_inbox_done"
term_data_dir = data_dir + "/term_data"
vector_storage_dir = data_dir + "/vector_index"
collection = vector_storage_dir.replace("/", "_").replace("_", "").replace(".", "")
doc_sum_index_dir = data_dir + "/doc_sum_index"
kg_graph_index_dir = data_dir + "/kg_graph_index"
aim_dir = data_dir + "/aim"
