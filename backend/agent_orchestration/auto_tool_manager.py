"""
Advanced Auto-Discovery Tool Manager for AI agents.

This module provides an advanced tool manager that can automatically discover
and register tools from service objects using reflection and metadata.
"""

import inspect
import logging
from typing import List, Dict, Any, Optional, Union, Callable, Type, get_type_hints
from dataclasses import dataclass
from abc import ABC, abstractmethod
import asyncio

from langchain_core.tools import Tool

logger = logging.getLogger(__name__)

@dataclass
class ToolMetadata:
    """Metadata for a tool."""
    name: str
    description: str
    category: str
    service: str
    method_name: str
    examples: List[str] = None
    parameters: Dict[str, Any] = None
    return_type: Any = None
    is_async: bool = False

class ToolDiscoveryStrategy(ABC):
    """Abstract base class for tool discovery strategies."""
    
    @abstractmethod
    def discover_tools(self, service: Any, service_name: str) -> List[ToolMetadata]:
        """Discover tools from a service object."""
        pass

class MetadataBasedDiscovery(ToolDiscoveryStrategy):
    """Discover tools based on predefined metadata."""
    
    def __init__(self, tool_metadata: Dict[str, Dict[str, Any]]):
        self.tool_metadata = tool_metadata
    
    def discover_tools(self, service: Any, service_name: str) -> List[ToolMetadata]:
        """Discover tools using predefined metadata."""
        tools = []
        
        if service_name not in self.tool_metadata:
            return tools
        
        service_metadata = self.tool_metadata[service_name]
        
        for method_name, tool_info in service_metadata.items():
            if hasattr(service, method_name):
                method = getattr(service, method_name)
                if callable(method):
                    # Get method signature
                    sig = inspect.signature(method)
                    is_async = inspect.iscoroutinefunction(method)
                    
                    # Extract parameter information
                    parameters = {}
                    for param_name, param in sig.parameters.items():
                        if param_name != 'self':
                            parameters[param_name] = {
                                'type': param.annotation if param.annotation != inspect.Parameter.empty else Any,
                                'default': param.default if param.default != inspect.Parameter.empty else None,
                                'required': param.default == inspect.Parameter.empty
                            }
                    
                    tool = ToolMetadata(
                        name=tool_info['name'],
                        description=tool_info['description'],
                        category=tool_info.get('category', 'general'),
                        service=service_name,
                        method_name=method_name,
                        examples=tool_info.get('examples', []),
                        parameters=parameters,
                        return_type=sig.return_annotation if sig.return_annotation != inspect.Signature.empty else Any,
                        is_async=is_async
                    )
                    tools.append(tool)
        
        return tools

class ReflectionBasedDiscovery(ToolDiscoveryStrategy):
    """Discover tools by analyzing service methods using reflection."""
    
    def __init__(self, include_patterns: List[str] = None, exclude_patterns: List[str] = None):
        self.include_patterns = include_patterns or ['get_', 'create_', 'add_', 'update_', 'delete_', 'send_', 'find_']
        self.exclude_patterns = exclude_patterns or ['_', '__', 'private_']
    
    def discover_tools(self, service: Any, service_name: str) -> List[ToolMetadata]:
        """Discover tools by analyzing service methods."""
        tools = []
        
        # Get all methods from the service
        methods = inspect.getmembers(service, predicate=inspect.ismethod)
        
        for method_name, method in methods:
            # Skip private methods and special methods
            if any(pattern in method_name for pattern in self.exclude_patterns):
                continue
            
            # Check if method matches include patterns
            if not any(method_name.startswith(pattern) for pattern in self.include_patterns):
                continue
            
            # Skip methods that are likely internal
            if method_name in ['__init__', '__del__', 'authenticate', 'initialize_service']:
                continue
            
            # Analyze method signature
            sig = inspect.signature(method)
            is_async = inspect.iscoroutinefunction(method)
            
            # Extract parameter information
            parameters = {}
            for param_name, param in sig.parameters.items():
                if param_name != 'self':
                    parameters[param_name] = {
                        'type': param.annotation if param.annotation != inspect.Parameter.empty else Any,
                        'default': param.default if param.default != inspect.Parameter.empty else None,
                        'required': param.default == inspect.Parameter.empty
                    }
            
            # Generate description from method name
            description = self._generate_description(method_name, parameters)
            
            # Determine category based on method name
            category = self._determine_category(method_name, service_name)
            
            tool = ToolMetadata(
                name=method_name,
                description=description,
                category=category,
                service=service_name,
                method_name=method_name,
                parameters=parameters,
                return_type=sig.return_annotation if sig.return_annotation != inspect.Signature.empty else Any,
                is_async=is_async
            )
            tools.append(tool)
        
        return tools
    
    def _generate_description(self, method_name: str, parameters: Dict[str, Any]) -> str:
        """Generate a description for a method based on its name and parameters."""
        # Convert method name to readable description
        words = method_name.replace('_', ' ').split()
        
        # Capitalize and join words
        action = ' '.join(word.capitalize() for word in words)
        
        # Add parameter context if available
        if parameters:
            param_names = list(parameters.keys())
            if param_names:
                if len(param_names) == 1:
                    action += f" with {param_names[0]}"
                else:
                    action += f" with {', '.join(param_names[:-1])} and {param_names[-1]}"
        
        return action
    
    def _determine_category(self, method_name: str, service_name: str) -> str:
        """Determine the category of a method based on its name and service."""
        # Service-based categories
        service_categories = {
            'calendar': 'calendar',
            'gmail': 'communication',
            'tasks': 'productivity',
            'drive': 'storage',
            'sheets': 'data',
            'maps': 'location'
        }
        
        # Method-based categories
        method_categories = {
            'get_': 'retrieval',
            'create_': 'creation',
            'add_': 'creation',
            'update_': 'modification',
            'delete_': 'deletion',
            'send_': 'communication',
            'find_': 'search',
            'search_': 'search'
        }
        
        # Check service category first
        if service_name in service_categories:
            return service_categories[service_name]
        
        # Check method category
        for prefix, category in method_categories.items():
            if method_name.startswith(prefix):
                return category
        
        return 'general'

class AutoToolManager:
    """Advanced tool manager with automatic discovery capabilities."""
    
    def __init__(self, discovery_strategies: List[ToolDiscoveryStrategy] = None):
        self.discovery_strategies = discovery_strategies or []
        self.tools: List[Tool] = []
        self.tool_metadata: List[ToolMetadata] = []
        self.services: Dict[str, Any] = {}
    
    def add_discovery_strategy(self, strategy: ToolDiscoveryStrategy) -> None:
        """Add a tool discovery strategy."""
        self.discovery_strategies.append(strategy)
    
    def register_service(self, service_name: str, service: Any) -> None:
        """Register a service for tool discovery."""
        self.services[service_name] = service
    
    def discover_tools(self) -> List[ToolMetadata]:
        """Discover tools from all registered services using all strategies."""
        all_tools = []
        
        for service_name, service in self.services.items():
            for strategy in self.discovery_strategies:
                try:
                    tools = strategy.discover_tools(service, service_name)
                    all_tools.extend(tools)
                    logger.debug(f"Discovered {len(tools)} tools from {service_name} using {strategy.__class__.__name__}")
                except Exception as e:
                    logger.error(f"Error discovering tools from {service_name} using {strategy.__class__.__name__}: {e}")
        
        self.tool_metadata = all_tools
        return all_tools
    
    def create_langchain_tools(self) -> List[Tool]:
        """Create LangChain Tool objects from discovered tool metadata."""
        tools = []
        
        for metadata in self.tool_metadata:
            service = self.services[metadata.service]
            method = getattr(service, metadata.method_name)
            
            # Create a universal wrapper that handles any parameter pattern
            def create_universal_wrapper(service_method, method_metadata):
                async def async_wrapper(**kwargs):
                    return await self._call_service_method(service_method, method_metadata, kwargs)
                
                def sync_wrapper(**kwargs):
                    return self._call_service_method(service_method, method_metadata, kwargs)
                
                # Return the appropriate wrapper based on whether the method is async
                if asyncio.iscoroutinefunction(service_method):
                    return async_wrapper
                else:
                    return sync_wrapper
            
            # Create the universal wrapper function
            wrapped_method = create_universal_wrapper(method, metadata)
            
            tool = Tool(
                name=metadata.name,
                func=wrapped_method,
                description=metadata.description
            )
            tools.append(tool)
        
        self.tools = tools
        return tools
    
    def _call_service_method(self, method: Callable, metadata: ToolMetadata, kwargs: Dict[str, Any]):
        """
        Universal method caller that adapts arguments to match the method signature.
        
        This method analyzes the target method's signature and adapts the incoming
        keyword arguments to match the expected parameter pattern.
        """
        import inspect
        
        # Get the method signature
        sig = inspect.signature(method)
        parameters = list(sig.parameters.values())
        
        # Skip 'self' parameter for instance methods
        if parameters and parameters[0].name == 'self':
            parameters = parameters[1:]
        
        if not parameters:
            # Method takes no parameters
            return method()
        
        if len(parameters) == 1:
            # Single parameter - could be a dictionary or any type
            param = parameters[0]
            param_name = param.name
            
            if param.annotation == inspect.Parameter.empty:
                # No type annotation - try to be smart about it
                if param_name.lower() in ['args', 'event_details', 'data', 'body', 'payload']:
                    # Likely expects a dictionary
                    return method(kwargs)
                else:
                    # Try to pass as single argument or unpack
                    if param_name in kwargs:
                        return method(kwargs[param_name])
                    else:
                        return method(kwargs)
            else:
                # Has type annotation
                if hasattr(param.annotation, '__origin__') and param.annotation.__origin__ is dict:
                    # Expects a dictionary
                    return method(kwargs)
                elif param.annotation == str and len(kwargs) == 1:
                    # Expects a string - try to combine or use the first value
                    first_key = next(iter(kwargs))
                    if isinstance(kwargs[first_key], str):
                        return method(kwargs[first_key])
                    else:
                        return method(str(kwargs[first_key]))
                else:
                    # Try to pass as single argument
                    if param_name in kwargs:
                        return method(kwargs[param_name])
                    else:
                        return method(kwargs)
        else:
            # Multiple parameters - use keyword arguments
            # Filter kwargs to only include parameters that the method expects
            filtered_kwargs = {}
            for param in parameters:
                if param.name in kwargs:
                    filtered_kwargs[param.name] = kwargs[param.name]
                elif param.default != inspect.Parameter.empty:
                    # Use default value
                    filtered_kwargs[param.name] = param.default
                else:
                    # Required parameter missing - try to provide a reasonable default
                    if param.annotation == str:
                        filtered_kwargs[param.name] = ""
                    elif param.annotation == int:
                        filtered_kwargs[param.name] = 0
                    elif param.annotation == bool:
                        filtered_kwargs[param.name] = False
                    elif hasattr(param.annotation, '__origin__') and param.annotation.__origin__ is list:
                        filtered_kwargs[param.name] = []
                    elif hasattr(param.annotation, '__origin__') and param.annotation.__origin__ is dict:
                        filtered_kwargs[param_name] = {}
                    else:
                        filtered_kwargs[param.name] = None
            
            return method(**filtered_kwargs)
    
    def get_tools_by_category(self, category: str) -> List[Tool]:
        """Get tools by category."""
        category_tools = []
        for i, metadata in enumerate(self.tool_metadata):
            if metadata.category == category:
                category_tools.append(self.tools[i])
        return category_tools
    
    def get_tool_metadata(self, tool_name: str) -> Optional[ToolMetadata]:
        """Get metadata for a specific tool."""
        for metadata in self.tool_metadata:
            if metadata.name == tool_name:
                return metadata
        return None
    
    def validate_tools(self) -> List[str]:
        """Validate discovered tools and return any issues."""
        issues = []
        
        # Check for duplicate tool names
        tool_names = [metadata.name for metadata in self.tool_metadata]
        duplicates = [name for name in set(tool_names) if tool_names.count(name) > 1]
        if duplicates:
            issues.append(f"Duplicate tool names found: {duplicates}")
        
        # Check for tools without descriptions
        for metadata in self.tool_metadata:
            if not metadata.description or metadata.description.strip() == '':
                issues.append(f"Tool '{metadata.name}' has no description")
        
        # Validate tool signatures
        for metadata in self.tool_metadata:
            try:
                service = self.services[metadata.service]
                method = getattr(service, metadata.method_name)
                
                # Test the universal wrapper with sample arguments
                test_kwargs = {"test_param": "test_value"}
                try:
                    # Create a test wrapper
                    def test_wrapper(**kwargs):
                        return self._call_service_method(method, metadata, kwargs)
                    
                    # Try calling with test arguments
                    test_wrapper(**test_kwargs)
                except Exception as e:
                    issues.append(f"Tool '{metadata.name}' signature validation failed: {str(e)}")
                    
            except Exception as e:
                issues.append(f"Tool '{metadata.name}' validation error: {str(e)}")
        
        return issues

    def get_tool_signature_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed signature information for a tool.
        
        This helps with debugging and understanding how to call tools correctly.
        """
        metadata = self.get_tool_metadata(tool_name)
        if not metadata:
            return None
        
        try:
            service = self.services[metadata.service]
            method = getattr(service, metadata.method_name)
            
            import inspect
            sig = inspect.signature(method)
            
            signature_info = {
                "name": tool_name,
                "service": metadata.service,
                "method": metadata.method_name,
                "is_async": metadata.is_async,
                "parameters": {},
                "return_type": str(sig.return_annotation) if sig.return_annotation != inspect.Signature.empty else "Any"
            }
            
            for param_name, param in sig.parameters.items():
                if param_name != 'self':
                    signature_info["parameters"][param_name] = {
                        "type": str(param.annotation) if param.annotation != inspect.Parameter.empty else "Any",
                        "default": param.default if param.default != inspect.Parameter.empty else None,
                        "required": param.default == inspect.Parameter.empty
                    }
            
            return signature_info
            
        except Exception as e:
            logger.error(f"Error getting signature info for {tool_name}: {e}")
            return None
    
    def generate_tool_documentation(self) -> str:
        """Generate documentation for all discovered tools."""
        doc_lines = ["# Tool Documentation", ""]
        
        # Group by category
        categories = {}
        for metadata in self.tool_metadata:
            if metadata.category not in categories:
                categories[metadata.category] = []
            categories[metadata.category].append(metadata)
        
        for category, tools in categories.items():
            doc_lines.append(f"## {category.title()}", "")
            
            for tool in tools:
                doc_lines.append(f"### {tool.name}", "")
                doc_lines.append(f"**Description:** {tool.description}", "")
                doc_lines.append(f"**Service:** {tool.service}", "")
                doc_lines.append(f"**Method:** {tool.method_name}", "")
                doc_lines.append(f"**Async:** {tool.is_async}", "")
                
                if tool.parameters:
                    doc_lines.append("**Parameters:**", "")
                    for param_name, param_info in tool.parameters.items():
                        required = "required" if param_info['required'] else "optional"
                        doc_lines.append(f"- `{param_name}` ({param_info['type'].__name__}, {required})", "")
                
                if tool.examples:
                    doc_lines.append("**Examples:**", "")
                    for example in tool.examples:
                        doc_lines.append(f"- {example}", "")
                
                doc_lines.append("", "")
        
        return "\n".join(doc_lines) 