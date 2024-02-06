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
    elif llm_engine == "ollama":
        from llama_index.llms import Ollama
        api_base_url = f"http://{host_ip}:11434"
        print(f"About to instanciate LLM {llm_model} on {api_base_url} using Ollama ...")
        return Ollama(
            model=llm_model, 
            base_url=api_base_url, 
            request_timeout=900, 
            temperature=temperature,
            additional_kwargs={"num_predict": 512}
            #additional_kwargs={"main_gpu": 1} # see https://github.com/jmorganca/ollama/issues/1813#issuecomment-1902682612
        )
    elif llm_engine == "ollama-gpu0":
        # Needs an Ollama-instance starting with this command: "CUDA_VISIBLE_DEVICES=0 OLLAMA_HOST=0.0.0.0:11535 ollama serve"
        from llama_index.llms import Ollama
        api_base_url = f"http://{host_ip}:11430"
        print(f"About to instanciate LLM {llm_model} on {api_base_url} using Ollama-Instance with GPU-ID 0 ...")
        return Ollama(model=llm_model, base_url=api_base_url, request_timeout=900, temperature=temperature, additional_kwargs={"num_predict": 512})
    elif llm_engine == "ollama-gpu1":
        # Needs an Ollama-instance starting with this command: "CUDA_VISIBLE_DEVICES=0 OLLAMA_HOST=0.0.0.0:11535 ollama serve"
        from llama_index.llms import Ollama
        api_base_url = f"http://{host_ip}:11431"
        print(f"About to instanciate LLM {llm_model} on {api_base_url} using Ollama-Instance with GPU-ID 1 ...")
        return Ollama(model=llm_model, base_url=api_base_url, request_timeout=900, temperature=temperature, additional_kwargs={"num_predict": 512})
    else:
        raise Exception(f"Unknown llm_engine: {llm_engine}")
    


communication_log_csv = "/data/llm_responses.csv"
def get_csv_callback_handler():
    from lib.callbacks.simple_dict_collector import SimpleDictStoreHandler
    return SimpleDictStoreHandler(communication_log_csv)

aim_callback = None
def get_aim_callback(aim_experiment_name, aim_path, aim_run_params: Optional[Dict[str, Any]] = None):
    global aim_callback
    from llama_index.callbacks import AimCallback
    if aim_callback is None:
        aim_callback = AimCallback(experiment_name=aim_experiment_name, repo=aim_path, run_params=aim_run_params)
    return aim_callback

callback_manager = None
def get_callback_manager(aim_path, aim_run_params: Optional[Dict[str, Any]] = None):
    global callback_manager
    if callback_manager is None:
        from llama_index.callbacks.base import CallbackManager
        callback_manager = CallbackManager(handlers=[get_csv_callback_handler(), get_aim_callback(aim_path, aim_run_params)])
    return callback_manager
    
