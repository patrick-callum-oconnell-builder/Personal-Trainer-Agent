from typing import List, Optional, Any, Dict, Annotated
from langchain_core.messages import BaseMessage
import asyncio
from dataclasses import dataclass, field
import operator

from backend.dictionary_state import DictionaryState

def last(left, right):
    return right

@dataclass
class AgentState(DictionaryState):
    messages: Annotated[List[BaseMessage], operator.add] = field(default_factory=list)
    status: Annotated[str, last] = "active"
    missing_fields: Annotated[List[str], operator.add] = field(default_factory=list)
    last_tool_result: Annotated[Any, last] = None

    def __post_init__(self):
        self._validate_messages(self.messages)
        self._validate_status(self.status)
        self._validate_missing_fields(self.missing_fields)

    def _validate_messages(self, messages):
        if messages is not None and not isinstance(messages, list):
            raise ValueError("Messages must be a list")
        if messages is not None and not all(isinstance(m, BaseMessage) for m in messages):
            raise ValueError("All messages must be BaseMessage instances")

    def _validate_status(self, status):
        valid_statuses = ["active", "awaiting_user", "awaiting_tool", "error", "done"]
        if status not in valid_statuses:
            raise ValueError(f"Status must be one of {valid_statuses}")

    def _validate_missing_fields(self, missing_fields):
        if missing_fields is not None and not isinstance(missing_fields, list):
            raise ValueError("Missing fields must be a list")

    async def update(self, **kwargs):
        """Thread-safe state update method with validation."""
        async with self._lock:
            for key, value in kwargs.items():
                if hasattr(self, key):
                    if key == 'messages':
                        self._validate_messages(value)
                    elif key == 'status':
                        self._validate_status(value)
                    elif key == 'missing_fields':
                        self._validate_missing_fields(value)
                    setattr(self, key, value)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'AgentState':
        """Create a state from a dictionary, ensuring messages are BaseMessage objects."""
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
        def to_message(obj):
            if isinstance(obj, BaseMessage):
                return obj
            if not isinstance(obj, dict):
                return obj
            role = obj.get('role')
            if role == 'user':
                return HumanMessage(**obj)
            elif role == 'assistant':
                return AIMessage(**obj)
            elif role == 'system':
                return SystemMessage(**obj)
            else:
                return BaseMessage(**obj)
        messages = d.get("messages", [])
        messages = [to_message(m) for m in messages]
        return cls(
            messages=messages,
            status=d.get("status", "active"),
            missing_fields=d.get("missing_fields", []),
            last_tool_result=d.get("last_tool_result"),
        ) 