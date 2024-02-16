from typing import Any, Dict, List, Optional, cast

from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.callbacks import CBEventType, EventPayload

from llama_index.core.callbacks import EventPayload
from lib.index.helper import cur_simple_date_time_sec

import pandas as pd
import os

class SimpleDictStoreHandler(BaseCallbackHandler):
    """Callback handler for printing llms inputs/outputs."""

    csv_columns = ["prompt", "completion", "time", "messages_str", "response"]

    def __init__(self, store_file_csv: str) -> None:
        self.store_file_csv = store_file_csv
        super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])

    def _write_llm_event(self, payload: dict) -> None:
        from llama_index.core.llms import ChatMessage
        print("_write_llm_event" + str(payload.keys()))

        data = {"time": cur_simple_date_time_sec()}
        if EventPayload.PROMPT in payload:
            data["prompt"] = str(payload.get(EventPayload.PROMPT))
            data["completion"] = str(payload.get(EventPayload.COMPLETION))
        elif EventPayload.MESSAGES in payload:
            messages = cast(List[ChatMessage], payload.get(EventPayload.MESSAGES, []))
            data["messages_str"] = "\n".join([str(x) for x in messages])
            data["response"] = str(payload.get(EventPayload.RESPONSE))
        else:
            print("No data to write to store: " +payload.keys())

        self._append_to_store(data)

    def _append_to_store(self, data: dict) -> None:
        new = pd.DataFrame([data], columns=self.csv_columns)
        if os.path.exists(self.store_file_csv):
            new = pd.concat([new, pd.read_csv(self.store_file_csv)], ignore_index=True)
        new.to_csv(self.store_file_csv, index=False)

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        return

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        return
    
    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        return event_id

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Count the LLM or Embedding tokens as needed."""
        if event_type == CBEventType.LLM and payload is not None:
            self._write_llm_event(payload)
