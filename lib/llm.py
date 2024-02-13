import os
from time import sleep
from typing import Any, Callable, Dict, List, Optional, Sequence
from llama_index.llms import OpenAI
from llama_index.llms import Ollama
from llama_index.llms import OpenAILike
from lib import constants

def get_llm(llm_engine, llm_model, openai_model = None):
    temperature = 0.1
    if llm_engine == "together":
        if openai_model is None:
            raise Exception("openai_model must be set when using together.ai")
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
        print(f"About to instanciate LLM {openai_model} using OpenAI ...")
        return OpenAI(
            model=openai_model,
            #api_base=api_base_url,
            api_key=os.environ["OPENAI_API_KEY"],
            temperature=temperature
        )
    elif llm_engine == "ollama-multi":
        llm_urls = [
            f"http://{constants.host_ip_ollama}:"+get_port_for_ollama_variant("ollama-gpu1"),
            f"http://{constants.host_ip_ollama}:"+get_port_for_ollama_variant("ollama-gpu0"),
            f"http://{constants.host_ip}:"+get_port_for_ollama_variant("ollama")
         #   f"http://{constants.host_ip_ollama}:"+get_port_for_ollama_variant("ollama")
        ]
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

from llama_index.llms.base import llm_chat_callback, llm_completion_callback
from llama_index.llms.custom import CustomLLM
from llama_index.core.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
    MessageRole,
)
class MultiOllamaRoundRobin(CustomLLM):
    ollama_workers: List[Ollama]
    ollama_main: Ollama
    ollama_busy: List[bool]

    def __init__(self, ollama_workers: List[Ollama]):
        super().__init__(ollama_workers=ollama_workers, ollama_main=ollama_workers[0], ollama_busy = [False for _ in ollama_workers])

    def execute_on_free_worker(self, task_func: Callable):
        chosen_index = None
        attempts = 0
        while chosen_index is None:
            attempts += 1
            for i, busy in enumerate(self.ollama_busy):
                if not busy:
                    chosen_index = i
                    self.ollama_busy[i] = True
                    break
            if chosen_index is None:
                sleep(0.1)
        
        print(f" -- MultiOllamaRoundRobin --- --- --- --- --- --- --- --- --- --- -- Using Ollama-Instance {chosen_index} --- --- --- -- ATTEMPTS {attempts} --- --- ---")

        # Execute the task on the chosen worker
        try:
            response = task_func(self.ollama_workers[chosen_index])
        finally:
            self.ollama_busy[chosen_index] = False

        return response

    @llm_chat_callback()
    def chat(self, messages, **kwargs):
        return self.execute_on_free_worker(lambda worker: worker.chat(messages, **kwargs))
    
    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        return self.execute_on_free_worker(lambda worker: worker.stream_chat(messages, **kwargs))
    
    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        return self.execute_on_free_worker(lambda worker: worker.complete(prompt, formatted, **kwargs))
    
    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        return self.execute_on_free_worker(lambda worker: worker.stream_complete(prompt, formatted, **kwargs))
    
    @classmethod
    def class_name(cls) -> str:
        return "MultiOllamaRoundRobin"
    
    @property
    def metadata(self) -> LLMMetadata:
        """LLM metadata."""
        return self.ollama_main.metadata

def get_port_for_ollama_variant(llm_engine):
    if llm_engine == "ollama-gpu0":
        return "11430"
    elif llm_engine == "ollama-gpu1":
        return "11431"
    elif llm_engine == "ollama":
        return "11434"
    else:
        raise Exception(f"Unknown llm_engine: {llm_engine}. Known are 'ollama', 'ollama-gpu0', 'ollama-gpu1'")
    
def get_embed_model(embed_engine: str, embed_model_name: str):
    if embed_engine == "fastembed":
        from llama_index.embeddings.fastembed import FastEmbedEmbedding
        print(f"About to instanciate Embed Model {embed_model_name} using FastEmbedEmbedding ...")
        return FastEmbedEmbedding(model_name=embed_model_name, cache_dir=constants.embed_cache_dir)
    elif embed_engine.startswith("ollama"):
        api_base_url = f"http://{constants.host_ip}:{get_port_for_ollama_variant(embed_engine)}"
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
