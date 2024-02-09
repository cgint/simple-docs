import os
from typing import Any, Dict, Optional

host_ip = os.getenv("HOST_IP", "host.docker.internal")

def get_llm(llm_engine, llm_model, openai_model = None):
    temperature = 0.1
    if llm_engine == "together":
        if openai_model is None:
            raise Exception("openai_model must be set when using together.ai")
        from llama_index.llms import OpenAILike
        print(f"About to instanciate LLM {openai_model} using Together.ai ...")
        return OpenAILike(
            model=openai_model,
            api_base="https://api.together.xyz",
            api_key=os.getenv("TOGETHER_AI_KEY"),
            is_chat_model=True,
            is_function_calling_model=True,
            reuse_client=False, # When doing anything with large volumes of async API calls, setting this to false can improve stability.",
            max_retries=10,
            timeout=120,
            temperature=temperature
        )
    elif llm_engine == "openai":
        if openai_model is None:
            raise Exception("openai_model must be set when using OpenAI")
        from llama_index.llms import OpenAI
        print(f"About to instanciate LLM {openai_model} using OpenAI ...")
        return OpenAI(
            model=openai_model,
            #api_base=api_base_url,
            api_key=os.environ["OPENAI_API_KEY"],
            temperature=temperature
        )
    elif llm_engine.startswith("ollama"):
        from llama_index.llms import Ollama
        api_base_url = f"http://{host_ip}:{get_port_for_ollama_variant(llm_engine)}"
        print(f"About to instanciate LLM {llm_model} on {api_base_url} using Ollama-Instance {llm_engine} ...")
        return Ollama(
            model=llm_model, 
            base_url=api_base_url, 
            request_timeout=900, 
            temperature=temperature,
            additional_kwargs={"num_predict": 512}
        )
    else:
        raise Exception(f"Unknown llm_engine: {llm_engine}")
    
def get_port_for_ollama_variant(llm_engine):
    if llm_engine == "ollama-gpu0":
        return 11430
    elif llm_engine == "ollama-gpu1":
        return 11431
    elif llm_engine == "ollama":
        return 11434
    else:
        raise Exception(f"Unknown llm_engine: {llm_engine}. Known are 'ollama', 'ollama-gpu0', 'ollama-gpu1'")
    
def get_embed_model(embed_engine: str, embed_model_name: str):
    if embed_engine == "fastembed":
        from llama_index.embeddings.fastembed import FastEmbedEmbedding
        embed_cache_dir = "/data/fastembed_cache/"
        print(f"About to instanciate Embed Model {embed_model_name} using FastEmbedEmbedding ...")
        return FastEmbedEmbedding(model_name=embed_model_name, cache_dir=embed_cache_dir)
    elif embed_engine.startswith("ollama"):
        api_base_url = f"http://{host_ip}:{get_port_for_ollama_variant(embed_engine)}"
        from llama_index.embeddings.ollama_embedding import OllamaEmbedding
        print(f"About to instanciate Embed Model {embed_model_name} using OllamaEmbedding ...")
        return OllamaEmbedding(model_name=embed_model_name, base_url=api_base_url)
    else:
        raise Exception(f"Unknown embed_model_name: {embed_model_name}")

communication_log_csv = "/data/llm_responses.csv"
def get_csv_callback_handler():
    from lib.callbacks.simple_dict_collector import SimpleDictStoreHandler
    return SimpleDictStoreHandler(communication_log_csv)

def get_aim_callback(aim_experiment_name, aim_path, aim_run_params: Optional[Dict[str, Any]] = None):
    from llama_index.callbacks import AimCallback
    return AimCallback(experiment_name=aim_experiment_name, repo=aim_path, run_params=aim_run_params)

def get_callback_manager(aim_path, aim_run_params: Optional[Dict[str, Any]] = None):
    from llama_index.callbacks.base import CallbackManager
    return CallbackManager(handlers=[get_csv_callback_handler(), get_aim_callback(aim_path, aim_run_params)])
