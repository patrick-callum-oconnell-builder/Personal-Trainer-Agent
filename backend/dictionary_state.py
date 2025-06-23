import asyncio
from typing import Any, Dict
from dataclasses import dataclass, field


@dataclass
class DictionaryState:
    """
    Base class providing dictionary-like functionality with thread-safe operations.
    
    This class serves as a foundation for state objects that need to behave like
    dictionaries while providing thread-safe access and common utility methods.
    It can be inherited by any class that needs dictionary-like behavior with
    additional safety and convenience features.
    
    Attributes:
        _lock (asyncio.Lock): Thread lock for ensuring thread-safe operations.
            Automatically created for each instance.
    
    Example:
        >>> class MyState(DictionaryState):
        ...     name: str = "default"
        ...     count: int = 0
        >>> state = MyState()
        >>> state['name'] = "example"
        >>> print(state['name'])
        example
        >>> 'count' in state
        True
        >>> state.get('nonexistent', 'default')
        'default'
    """
    
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    
    def __getitem__(self, key: str) -> Any:
        """
        Get an item using dictionary-style access.
        
        Args:
            key (str): The attribute name to retrieve.
            
        Returns:
            Any: The value of the specified attribute.
            
        Raises:
            KeyError: If the attribute doesn't exist.
            
        Example:
            >>> state = MyState()
            >>> value = state['attribute_name']
        """
        if not hasattr(self, key):
            raise KeyError(f"{self.__class__.__name__} has no attribute '{key}'")
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Set an item using dictionary-style access.
        
        Args:
            key (str): The attribute name to set.
            value (Any): The value to assign to the attribute.
            
        Raises:
            KeyError: If the attribute doesn't exist.
            
        Example:
            >>> state = MyState()
            >>> state['attribute_name'] = "new_value"
        """
        if not hasattr(self, key):
            raise KeyError(f"{self.__class__.__name__} has no attribute '{key}'")
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:
        """
        Check if a key exists in the state.
        
        Args:
            key (str): The attribute name to check.
            
        Returns:
            bool: True if the attribute exists, False otherwise.
            
        Example:
            >>> state = MyState()
            >>> 'attribute_name' in state
            True
        """
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get an item with a default value if the key doesn't exist.
        
        Args:
            key (str): The attribute name to retrieve.
            default (Any, optional): Default value to return if attribute doesn't exist.
                Defaults to None.
                
        Returns:
            Any: The value of the specified attribute or the default value.
            
        Example:
            >>> state = MyState()
            >>> value = state.get('nonexistent', 'default_value')
            >>> print(value)
            default_value
        """
        return getattr(self, key, default)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the state to a dictionary.
        
        Only public attributes (those not starting with '_') are included
        in the resulting dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the state.
            
        Example:
            >>> state = MyState(name="test", count=42)
            >>> state_dict = state.to_dict()
            >>> print(state_dict)
            {'name': 'test', 'count': 42}
        """
        result = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_'):  # Skip private attributes
                result[key] = value
        return result

    async def update(self, **kwargs) -> None:
        """
        Thread-safe state update method.
        
        Updates multiple attributes atomically using the instance's lock.
        Only updates attributes that already exist on the instance.
        
        Args:
            **kwargs: Keyword arguments where keys are attribute names and
                     values are the new values to assign.
                     
        Example:
            >>> await state.update(name="new_name", count=100)
        """
        async with self._lock:
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(self, key, value)

    def __repr__(self) -> str:
        """
        String representation of the state.
        
        Returns a formatted string showing the class name and all public
        attributes. For list attributes, shows the length instead of the
        full content to avoid overly long representations.
        
        Returns:
            str: String representation of the state.
            
        Example:
            >>> state = MyState(name="test", items=[1, 2, 3])
            >>> print(repr(state))
            MyState(name=test, items=3)
        """
        attrs = []
        for key, value in self.__dict__.items():
            if not key.startswith('_'):  # Skip private attributes
                if isinstance(value, list):
                    attrs.append(f"{key}={len(value)}")
                else:
                    attrs.append(f"{key}={value}")
        return f"{self.__class__.__name__}({', '.join(attrs)})"

    def __eq__(self, other: Any) -> bool:
        """
        Compare two states for equality.
        
        Compares all public attributes (those not starting with '_') between
        this instance and another instance of the same class.
        
        Args:
            other (Any): The object to compare with.
            
        Returns:
            bool: True if both objects have the same class and all public
                  attributes are equal, False otherwise.
                  
        Example:
            >>> state1 = MyState(name="test", count=42)
            >>> state2 = MyState(name="test", count=42)
            >>> state1 == state2
            True
        """
        if not isinstance(other, self.__class__):
            return False
        
        # Compare all non-private attributes
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                if not hasattr(other, key) or getattr(other, key) != value:
                    return False
        return True 