import asyncio
from typing import Any, Dict
from dataclasses import dataclass, field


@dataclass
class DictionaryState:
    """Base class providing dictionary-like functionality with thread-safe operations."""
    
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    
    def __getitem__(self, key: str) -> Any:
        """Get an item using dictionary-style access."""
        if not hasattr(self, key):
            raise KeyError(f"{self.__class__.__name__} has no attribute '{key}'")
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Set an item using dictionary-style access."""
        if not hasattr(self, key):
            raise KeyError(f"{self.__class__.__name__} has no attribute '{key}'")
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        """Check if a key exists in the state."""
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        """Get an item with a default value if the key doesn't exist."""
        return getattr(self, key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the state to a dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_'):  # Skip private attributes
                result[key] = value
        return result

    async def update(self, **kwargs):
        """Thread-safe state update method."""
        async with self._lock:
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)

    def __repr__(self) -> str:
        """String representation of the state."""
        attrs = []
        for key, value in self.__dict__.items():
            if not key.startswith('_'):  # Skip private attributes
                if isinstance(value, list):
                    attrs.append(f"{key}={len(value)}")
                else:
                    attrs.append(f"{key}={value}")
        return f"{self.__class__.__name__}({', '.join(attrs)})"

    def __eq__(self, other: Any) -> bool:
        """Compare two states for equality."""
        if not isinstance(other, self.__class__):
            return False
        
        # Compare all non-private attributes
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                if not hasattr(other, key) or getattr(other, key) != value:
                    return False
        return True 