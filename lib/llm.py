import os
from typing import Any, Dict, Optional
from llama_index.llms.openai_like import OpenAILike
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.llms.groq import Groq
from llama_index.llms.gemini import Gemini
from lib import constants
from lib.llm_fallback import MultiLlmFallback
from lib.llm_round_robin import MultiOllamaRoundRobin

def get_llm_multi(llm_urls, llm_engine, llm_model, temperature, openai_model = None):
        print(f"About to instanciate LLM {llm_model} on {llm_urls} using Ollama-Instance {llm_engine} ...")
        workers = [
            Ollama(
                model=llm_model, 
                base_url=llm_url, 
                request_timeout=900, 
                temperature=temperature,
                additional_kwargs={"num_predict": 1024}
            )
            for llm_url in llm_urls
        ]
        return MultiOllamaRoundRobin(workers)
safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_ONLY_HIGH",
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_ONLY_HIGH",
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_ONLY_HIGH",
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_ONLY_HIGH",
  }
]
def get_gemini(llm_model, temperature):
    generation_config = {
        "temperature": temperature,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 16000,
        "response_mime_type": "text/plain"
    }
    return Gemini(model_name=llm_model, generation_config=generation_config, safety_settings=safety_settings, api_key=os.environ.get("GEMINI_API_KEY"))

def get_together(openai_model, temperature):
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

def get_groq(llm_model):
     return Groq(model=llm_model, api_key=os.environ.get("GROQ_API_KEY"))

def get_llm(llm_engine, llm_model, openai_model = None):
    temperature = 0.1
    if llm_engine == "gemini":
        print(f"About to instanciate LLM {llm_model} using Gemini ...")
        return get_gemini(llm_model, temperature)
    elif llm_engine == "together":
        if openai_model is None:
            raise Exception("openai_model must be set when using together.ai")
        print(f"About to instanciate LLM {openai_model} using Together.ai ...")
        return get_together(openai_model, temperature)
    elif llm_engine == "openai":
        if openai_model is None:
            raise Exception("openai_model must be set when using OpenAI")
        print(f"About to instanciate LLM {openai_model} using OpenAI ...")
        return OpenAI(
            model=openai_model,
            #api_base=api_base_url,
            api_key=os.environ["OPENAI_API_KEY"],
            temperature=temperature
        )
    elif llm_engine == "groq":
        print(f"About to instanciate LLM {openai_model} using Groq-Cloud ...")
        return get_groq(llm_model)
    elif llm_engine == "groq-together":
        print(f"About to instanciate Groq with {llm_model} and as fallback Together with {openai_model} ...")
        return MultiLlmFallback([
             get_groq(llm_model),
             get_together(openai_model, temperature)
        ])
    elif llm_engine == "ollama-multi":
        llm_urls = [
            f"http://{constants.host_ip_ollama}:"+get_port_for_ollama_variant("ollama-gpu1"),
            f"http://{constants.host_ip_ollama}:"+get_port_for_ollama_variant("ollama-gpu0"),
            f"http://{constants.host_ip}:"+get_port_for_ollama_variant("ollama")
        ]
        return get_llm_multi(llm_urls, llm_engine, llm_model, temperature, openai_model)
    elif llm_engine == "ollama-multi-local-2":
        llm_urls = [
            f"http://{constants.host_ip_ollama}:"+get_port_for_ollama_variant("ollama-gpu1"),
            f"http://{constants.host_ip_ollama}:"+get_port_for_ollama_variant("ollama-gpu0")
        ]
        return get_llm_multi(llm_urls, llm_engine, llm_model, temperature, openai_model)
    elif llm_engine == "ollama-multi-local-4":
        llm_urls = [
            f"http://{constants.host_ip}:11431",
            f"http://{constants.host_ip}:11432",
            f"http://{constants.host_ip}:11433",
            f"http://{constants.host_ip}:11434",
        ]
        return get_llm_multi(llm_urls, llm_engine, llm_model, temperature, openai_model)
    elif llm_engine.startswith("ollama"):
        api_base_url = f"http://{constants.host_ip_ollama}:{get_port_for_ollama_variant(llm_engine)}"
        print(f"About to instanciate LLM {llm_model} on {api_base_url} using Ollama-Instance {llm_engine} ...")
        return Ollama(
            model=llm_model, 
            base_url=api_base_url, 
            request_timeout=900, 
            temperature=temperature,
            additional_kwargs={"num_predict": 1024}
        )
    else:
        raise Exception(f"Unknown llm_engine: {llm_engine}")

def get_port_for_ollama_variant(llm_engine):
    if llm_engine == "ollama-gpu0":
        return "11431"
    elif llm_engine == "ollama-gpu1":
        return "11432"
    elif llm_engine == "ollama":
        return "11434"
    elif llm_engine == "ollama-ssh":
        return "11400"
    else:
        raise Exception(f"Unknown llm_engine: {llm_engine}. Known are 'ollama', 'ollama-gpu0', 'ollama-gpu1'")
    
def get_embed_model(embed_engine: str, embed_model_name: str):
    from llama_index.embeddings.fastembed import FastEmbedEmbedding
    from llama_index.embeddings.ollama import OllamaEmbedding
    if embed_engine == "fastembed":
        print(f"About to instanciate Embed Model {embed_model_name} using FastEmbedEmbedding ...")
        return FastEmbedEmbedding(model_name=embed_model_name, cache_dir=constants.embed_cache_dir)
    elif embed_engine.startswith("ollama"):
        api_base_url = f"http://{constants.host_ip}:{get_port_for_ollama_variant(embed_engine)}"
        print(f"About to instanciate Embed Model {embed_model_name} using OllamaEmbedding ...")
        return OllamaEmbedding(model_name=embed_model_name, base_url=api_base_url)
    else:
        raise Exception(f"Unknown embed_model_name: {embed_model_name}")

communication_log_csv = "/data/llm_responses.csv"
def get_csv_callback_handler():
    from lib.callbacks.simple_dict_collector import SimpleDictStoreHandler
    return SimpleDictStoreHandler(communication_log_csv)

def get_aim_callback(aim_experiment_name, aim_path, aim_run_params: Optional[Dict[str, Any]] = None):
    from llama_index.callbacks.aim import AimCallback
    return AimCallback(experiment_name=aim_experiment_name, repo=aim_path, run_params=aim_run_params)

def get_callback_manager(aim_path, aim_run_params: Optional[Dict[str, Any]] = None):
    from llama_index.core.callbacks import CallbackManager
    return CallbackManager(handlers=[get_csv_callback_handler(), get_aim_callback(aim_path, aim_run_params)])
