from typing import List, Optional, Any, Dict, Annotated
from langchain_core.messages import BaseMessage
import asyncio
from dataclasses import dataclass, field
import operator
import logging

from backend.dictionary_state import DictionaryState

def last(left, right):
    """Return the rightmost value when merging states."""
    return right

# Class-level state history (outside the dataclass)
_state_history: List[Dict[str, Any]] = []

@dataclass
class AgentState(DictionaryState):
    """
    State management class for AI agent conversations and workflow.
    
    This class extends DictionaryState to provide specialized state management
    for AI agents, including message history, status tracking, and tool results.
    It includes validation for agent-specific data types and provides methods
    for safe state updates with validation.
    
    Attributes:
        messages (List[BaseMessage]): List of conversation messages between
            the user and agent. Uses operator.add for merging states.
        status (str): Current status of the agent. Valid values are:
            - "active": Agent is actively processing
            - "awaiting_user": Waiting for user input
            - "awaiting_tool": Waiting for tool execution
            - "error": Agent encountered an error
            - "done": Agent has completed its task
        missing_fields (List[str]): List of required fields that are missing
            from the current conversation context.
        last_tool_result (Any): Result from the most recently executed tool.
    
    Example:
        >>> from langchain_core.messages import HumanMessage, AIMessage
        >>> state = AgentState()
        >>> state.messages = [HumanMessage(content="Hello")]
        >>> state.status = "active"
        >>> print(state['status'])
        active
        >>> 'messages' in state
        True
    """
    
    messages: Annotated[List[BaseMessage], operator.add] = field(default_factory=list)
    status: Annotated[str, last] = "active"
    missing_fields: Annotated[List[str], operator.add] = field(default_factory=list)
    last_tool_result: Annotated[Any, last] = None

    def __post_init__(self):
        """
        Validate the initial state after object creation.
        
        Called automatically after the dataclass is initialized to ensure
        all attributes have valid values according to the validation rules.
        
        Raises:
            ValueError: If any validation fails.
        """
        self._validate_messages(self.messages)
        self._validate_status(self.status)
        self._validate_missing_fields(self.missing_fields)

    def _validate_messages(self, messages):
        """
        Validate that messages is a list of BaseMessage objects.
        
        Args:
            messages: The messages to validate.
            
        Raises:
            ValueError: If messages is not a list or contains non-BaseMessage objects.
        """
        if messages is not None and not isinstance(messages, list):
            raise ValueError("Messages must be a list")
        if messages is not None and not all(isinstance(m, BaseMessage) for m in messages):
            raise ValueError("All messages must be BaseMessage instances")

    def _validate_status(self, status):
        """
        Validate that status is one of the allowed values.
        
        Args:
            status: The status to validate.
            
        Raises:
            ValueError: If status is not one of the valid statuses.
        """
        valid_statuses = ["active", "awaiting_user", "awaiting_tool", "error", "done"]
        if status not in valid_statuses:
            raise ValueError(f"Status must be one of {valid_statuses}")

    def _validate_missing_fields(self, missing_fields):
        """
        Validate that missing_fields is a list.
        
        Args:
            missing_fields: The missing fields to validate.
            
        Raises:
            ValueError: If missing_fields is not a list.
        """
        if missing_fields is not None and not isinstance(missing_fields, list):
            raise ValueError("Missing fields must be a list")

    async def update(self, **kwargs):
        """
        Thread-safe state update method with validation.
        
        Updates the agent state with validation for agent-specific fields.
        Only updates attributes that exist and pass validation.
        
        Args:
            **kwargs: Keyword arguments where keys are attribute names and
                     values are the new values to assign.
                     
        Raises:
            ValueError: If any validation fails for the updated fields.
            
        Example:
            >>> await state.update(status="awaiting_user", missing_fields=["name"])
            >>> await state.update(messages=[HumanMessage(content="Hello")])
        """
        async with self._lock:
            logger = logging.getLogger(__name__)
            logger.info(f"Updating agent state with: {kwargs}")
            
            for key, value in kwargs.items():
                if hasattr(self, key):
                    if key == 'messages':
                        self._validate_messages(value)
                    elif key == 'status':
                        self._validate_status(value)
                    elif key == 'missing_fields':
                        self._validate_missing_fields(value)
                    setattr(self, key, value)
                    logger.debug(f"Updated {key} to {value}")
                else:
                    logger.warning(f"Attempted to update non-existent attribute: {key}")
            
            # After updating, append a snapshot to the state history
            self.append_to_history()

    def append_to_history(self):
        # Store a snapshot of the current state (excluding private fields)
        snapshot = self.to_dict()
        global _state_history
        _state_history.append(snapshot)
        logging.getLogger(__name__).info(f"State history updated. Total snapshots: {len(_state_history)}")
        logging.getLogger(__name__).debug(f"Current state: {snapshot}")

    @classmethod
    def get_state_history(cls) -> List[Dict[str, Any]]:
        """Return the state history as a list of dicts."""
        global _state_history
        return _state_history.copy()

    @classmethod
    def clear_state_history(cls):
        """Clear the state history. Useful for testing."""
        global _state_history
        _state_history.clear()

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'AgentState':
        """
        Create an AgentState instance from a dictionary.
        
        Converts dictionary data into a properly formatted AgentState object,
        ensuring that message objects are correctly instantiated from their
        dictionary representations.
        
        Args:
            d (Dict[str, Any]): Dictionary containing state data with keys:
                - messages: List of message dictionaries or BaseMessage objects
                - status: String status value
                - missing_fields: List of missing field names
                - last_tool_result: Any tool result value
                
        Returns:
            AgentState: A new AgentState instance with the provided data.
            
        Example:
            >>> data = {
            ...     "messages": [{"role": "user", "content": "Hello"}],
            ...     "status": "active",
            ...     "missing_fields": ["name"],
            ...     "last_tool_result": None
            ... }
            >>> state = AgentState.from_dict(data)
            >>> print(state.status)
            active
        """
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
        def to_message(obj):
            """Convert dictionary to appropriate message object."""
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