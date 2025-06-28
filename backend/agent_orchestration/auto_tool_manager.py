"""
Advanced Auto-Discovery Tool Manager for AI agents.

This module provides an advanced tool manager that can automatically discover
and register tools from service objects using reflection and metadata.
"""

import asyncio
import inspect
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union, Callable, Type, get_type_hints

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
                import logging
                def sync_wrapper(*args, **kwargs):
                    logger = logging.getLogger(__name__)
                    # Handle different calling patterns from LangChain
                    if args and kwargs:
                        # Both args and kwargs provided - combine them
                        all_args = {}
                        if isinstance(args[0], dict):
                            all_args.update(args[0])
                        else:
                            all_args["input"] = args[0]
                        all_args.update(kwargs)
                    elif args:
                        # Only args provided
                        if isinstance(args[0], dict):
                            all_args = args[0]
                        else:
                            all_args = {"input": args[0]}
                    else:
                        # Only kwargs provided
                        all_args = kwargs
                    
                    logger.debug(f"Calling {service_method.__name__} with args: {all_args}")
                    return self._call_service_method(service_method, method_metadata, all_args)
                
                async def async_wrapper(*args, **kwargs):
                    logger = logging.getLogger(__name__)
                    # Handle different calling patterns from LangChain
                    if args and kwargs:
                        # Both args and kwargs provided - combine them
                        all_args = {}
                        if isinstance(args[0], dict):
                            all_args.update(args[0])
                        else:
                            all_args["input"] = args[0]
                        all_args.update(kwargs)
                    elif args:
                        # Only args provided
                        if isinstance(args[0], dict):
                            all_args = args[0]
                        else:
                            all_args = {"input": args[0]}
                    else:
                        # Only kwargs provided
                        all_args = kwargs
                    
                    logger.debug(f"Calling {service_method.__name__} with args: {all_args}")
                    return await self._call_service_method_async(service_method, method_metadata, all_args)
                
                if asyncio.iscoroutinefunction(service_method):
                    return async_wrapper
                else:
                    return sync_wrapper
            wrapped_method = create_universal_wrapper(method, metadata)
            tool = Tool(
                name=metadata.name,
                func=wrapped_method,
                description=metadata.description
            )
            tools.append(tool)
        self.tools = tools
        return tools

    def _call_service_method(self, method: Callable, metadata: ToolMetadata, args: dict):
        """
        Universal method caller that adapts arguments to match the method signature.
        Always expects args as a dict.
        """
        import inspect
        sig = inspect.signature(method)
        parameters = list(sig.parameters.values())
        if parameters and parameters[0].name == 'self':
            parameters = parameters[1:]
        
        if not parameters:
            return method()
        elif len(parameters) == 1:
            # Single parameter: pass args as the single argument
            param = parameters[0]
            if param.kind in [inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD]:
                # Special handling for write_event method - convert individual parameters to event_details
                if metadata.method_name == "write_event" and "event_details" not in args:
                    # Convert individual event parameters to event_details format
                    event_details = {}
                    event_fields = ["summary", "description", "start", "end", "location", "attendees"]
                    for field in event_fields:
                        if field in args:
                            event_details[field] = args[field]
                    # Pass any remaining args that aren't event fields
                    for key, value in args.items():
                        if key not in event_fields:
                            event_details[key] = value
                    return method(event_details)
                elif metadata.method_name == "write_event" and "event_details" in args:
                    # Extract event_details from args
                    return method(args["event_details"])
                else:
                    # Pass the entire args dict as the single argument
                    return method(args)
            else:
                # For keyword-only parameters, unpack as kwargs
                return method(**args)
        else:
            # Multiple parameters: unpack args as kwargs
            return method(**args)
    
    async def _call_service_method_async(self, method: Callable, metadata: ToolMetadata, args: dict):
        """
        Async version of universal method caller that adapts arguments to match the method signature.
        Always expects args as a dict.
        """
        import inspect
        sig = inspect.signature(method)
        parameters = list(sig.parameters.values())
        if parameters and parameters[0].name == 'self':
            parameters = parameters[1:]
        
        if not parameters:
            return await method()
        elif len(parameters) == 1:
            # Single parameter: pass args as the single argument
            param = parameters[0]
            if param.kind in [inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD]:
                # Special handling for write_event method - convert individual parameters to event_details
                if metadata.method_name == "write_event" and "event_details" not in args:
                    # Convert individual event parameters to event_details format
                    event_details = {}
                    event_fields = ["summary", "description", "start", "end", "location", "attendees"]
                    for field in event_fields:
                        if field in args:
                            event_details[field] = args[field]
                    # Pass any remaining args that aren't event fields
                    for key, value in args.items():
                        if key not in event_fields:
                            event_details[key] = value
                    return await method(event_details)
                elif metadata.method_name == "write_event" and "event_details" in args:
                    # Extract event_details from args
                    return await method(args["event_details"])
                else:
                    # Pass the entire args dict as the single argument
                    return await method(args)
            else:
                # For keyword-only parameters, unpack as kwargs
                return await method(**args)
        else:
            # Multiple parameters: unpack args as kwargs
            return await method(**args)
    
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