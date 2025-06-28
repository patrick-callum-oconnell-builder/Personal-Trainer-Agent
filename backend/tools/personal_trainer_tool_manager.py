"""
Personal Trainer Tool Manager for AI agent.

This module contains the PersonalTrainerToolManager class that uses the
AutoToolManager for automatic tool discovery and registration.
"""

import json
import logging
import os
import asyncio
import pytz
import dateparser
import inspect
from datetime import datetime, timezone as dt_timezone, timedelta
from typing import List, Dict, Any, Optional, Union, Callable, Type

from langchain_core.tools import Tool
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from backend.agent_orchestration.auto_tool_manager import (
    AutoToolManager,
    MetadataBasedDiscovery,
    ReflectionBasedDiscovery
)
from backend.tools.tool_config import TOOL_METADATA, CUSTOM_TOOLS, TOOL_CATEGORIES
from backend.google_services import (
    GoogleCalendarService,
    GoogleDriveService,
    GoogleGmailService,
    GoogleMapsService,
    GoogleSheetsService,
    GoogleTasksService,
)
from backend.tools.preferences_tools import add_preference_to_kg
from backend.prompts import get_tool_result_summary_prompt
from backend.agent_orchestration.utilities import convert_natural_language_to_structured_args

logger = logging.getLogger(__name__)

# Load configuration
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.json')
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

TOOL_MANAGER_CONFIG = config['llm']['tool_manager']
DEFAULTS_CONFIG = config['defaults']

class PersonalTrainerToolManager:
    """
    Personal Trainer Tool Manager using Auto-Discovery.
    
    This class uses the AutoToolManager to automatically discover and register
    tools from Google services and custom functions.
    """
    
    def __init__(
        self,
        calendar_service: GoogleCalendarService,
        gmail_service: GoogleGmailService,
        tasks_service: GoogleTasksService,
        drive_service: GoogleDriveService,
        sheets_service: GoogleSheetsService,
        maps_service: Optional[GoogleMapsService] = None,
        llm: Optional[ChatOpenAI] = None
    ):
        # Initialize the auto tool manager
        self.auto_manager = AutoToolManager()
        
        # Add discovery strategies
        metadata_strategy = MetadataBasedDiscovery(TOOL_METADATA)
        reflection_strategy = ReflectionBasedDiscovery()
        self.auto_manager.add_discovery_strategy(metadata_strategy)
        self.auto_manager.add_discovery_strategy(reflection_strategy)
        
        # Store services in a dictionary for easier access
        self.services = {
            'calendar': calendar_service,
            'gmail': gmail_service,
            'tasks': tasks_service,
            'drive': drive_service,
            'sheets': sheets_service,
            'maps': maps_service
        }
        
        # Register services with the auto manager
        for service_name, service in self.services.items():
            if service:  # Only register if service is available
                self.auto_manager.register_service(service_name, service)
        
        # Initialize LLM
        if llm:
            self.llm = llm
        else:
            self.llm = ChatOpenAI(
                model=TOOL_MANAGER_CONFIG['model'],
                temperature=TOOL_MANAGER_CONFIG['temperature'],
                streaming=True,
                callbacks=[StreamingStdOutCallbackHandler()],
                timeout=30,  # 30 second timeout
                max_retries=2,  # Retry up to 2 times
                request_timeout=30,  # Request timeout
            )
        self.user_timezone = os.environ.get("USER_TIMEZONE", DEFAULTS_CONFIG['user_timezone'])
        
        # Discover and create tools
        self._discover_and_create_tools()
        
        # Register custom tools that don't belong to services
        self._register_custom_tools()
        
        # Register special tools that require custom logic
        self._register_special_tools()
        
        # Validate tools
        issues = self.auto_manager.validate_tools()
        if issues:
            logger.warning(f"Tool validation issues: {issues}")

    def _discover_and_create_tools(self) -> None:
        """Discover tools from services and create LangChain tools."""
        logger.info("Discovering tools from services...")
        
        # Discover tools using auto manager
        discovered_tools = self.auto_manager.discover_tools()
        logger.info(f"Discovered {len(discovered_tools)} tools from services")
        
        # Create LangChain tools
        self.tools = self.auto_manager.create_langchain_tools()
        logger.info(f"Created {len(self.tools)} LangChain tools")

    def _register_custom_tools(self) -> None:
        """Register custom tools that don't belong to a specific service."""
        logger.debug("Registering custom tools...")
        
        for tool_name, tool_info in CUSTOM_TOOLS.items():
            tool = Tool(
                name=tool_name,
                func=tool_info['func'],
                description=tool_info['description']
            )
            self.tools.append(tool)
            logger.debug(f"Registered custom tool: {tool_name}")

    def _register_special_tools(self) -> None:
        """Register special tools that require custom logic or multiple services."""
        logger.debug("Registering special tools...")
        
        # Calendar conflict resolution tool (requires custom logic)
        if self.services['calendar']:
            self.tools.append(
                Tool(
                    name="resolve_calendar_conflict",
                    func=self._resolve_calendar_conflict,
                    description="Resolve calendar conflicts by replacing, deleting, or skipping conflicting events"
                )
            )
            logger.debug("Registered special tool: resolve_calendar_conflict")

    def get_tools(self) -> List[Tool]:
        """Get all available tools."""
        return self.tools

    def get_tools_by_category(self, category: str) -> List[Tool]:
        """Get all tools in a specific category."""
        # Get tools from auto manager
        auto_tools = self.auto_manager.get_tools_by_category(category)
        
        # Add custom tools in the same category
        custom_tools = []
        for tool_name, tool_info in CUSTOM_TOOLS.items():
            if tool_info.get('category') == category:
                tool = self.get_tool_by_name(tool_name)
                if tool:
                    custom_tools.append(tool)
        
        # Add special tools in the same category
        special_tools = []
        if category == 'calendar':
            tool = self.get_tool_by_name('resolve_calendar_conflict')
            if tool:
                special_tools.append(tool)
        
        return auto_tools + custom_tools + special_tools

    def get_available_categories(self) -> List[str]:
        """Get all available tool categories."""
        categories = set()
        
        # Get categories from auto manager
        for metadata in self.auto_manager.tool_metadata:
            if metadata.category:
                categories.add(metadata.category)
        
        # Get categories from custom tools
        for tool_info in CUSTOM_TOOLS.values():
            if tool_info.get('category'):
                categories.add(tool_info['category'])
        
        # Add categories for special tools
        categories.add('calendar')  # For resolve_calendar_conflict
        
        return list(categories)

    def get_service_status(self) -> Dict[str, bool]:
        """Get the status of all services (available/not available)."""
        return {
            service_name: service is not None 
            for service_name, service in self.services.items()
        }

    def get_tool_by_name(self, tool_name: str) -> Optional[Tool]:
        """Get a tool by name."""
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        return None

    def validate_tool_signature(self, tool: Tool) -> bool:
        """Validate that a tool has the expected signature."""
        if not hasattr(tool, 'func') or not callable(tool.func):
            return False
        
        # Check if the function is async or sync
        sig = inspect.signature(tool.func)
        return True  # Basic validation - could be extended

    async def execute_tool(self, tool_name: str, args: Union[str, Dict[str, Any]]) -> Any:
        """Execute a tool by name with the given arguments."""
        tool = self.get_tool_by_name(tool_name)
        if not tool:
            logger.error(f"Tool '{tool_name}' not found.")
            return f"Error: Tool '{tool_name}' not found."
        
        try:
            # Log the tool execution for debugging
            logger.debug(f"Executing tool '{tool_name}' with args: {args}")
            
            # Handle string arguments by parsing them into the appropriate format
            if isinstance(args, str):
                parsed_args = await self._parse_and_convert_args(tool_name, args)
                logger.debug(f"Parsed args for '{tool_name}': {parsed_args}")
            else:
                parsed_args = args
            
            # Validate arguments against tool signature
            validation_result = await self._validate_tool_arguments(tool_name, parsed_args)
            if validation_result:
                logger.warning(f"Tool argument validation warnings for '{tool_name}': {validation_result}")
            
            # Call the tool with parsed arguments
            if callable(tool.func):
                result = await self._maybe_await(tool.func(**parsed_args))
                logger.debug(f"Tool '{tool_name}' executed successfully")
                
                # Summarize the result for user-friendly output
                try:
                    summarized_result = await self.summarize_tool_result(tool_name, result)
                    return summarized_result
                except Exception as e:
                    logger.warning(f"Failed to summarize result for tool '{tool_name}': {e}")
                    # Fall back to raw result if summarization fails
                    return result
            else:
                return f"Error: Tool '{tool_name}' is not callable."
                
        except Exception as e:
            logger.error(f"Error executing tool '{tool_name}': {e}")
            
            # Provide detailed error information for debugging
            error_info = await self._get_detailed_error_info(tool_name, parsed_args, e)
            return f"Error executing tool '{tool_name}': {e}\n\nDebug info: {error_info}"

    async def _validate_tool_arguments(self, tool_name: str, args: Dict[str, Any]) -> Optional[str]:
        """
        Validate tool arguments against the expected signature.
        
        Returns a warning message if there are issues, None if everything is OK.
        """
        try:
            tool_metadata = self.auto_manager.get_tool_metadata(tool_name)
            if not tool_metadata or not tool_metadata.parameters:
                return None
            
            warnings = []
            
            # Check for missing required parameters
            for param_name, param_info in tool_metadata.parameters.items():
                if param_info.get('required', True) and param_name not in args:
                    warnings.append(f"Missing required parameter: {param_name}")
            
            # Check for unexpected parameters
            expected_params = set(tool_metadata.parameters.keys())
            provided_params = set(args.keys())
            unexpected_params = provided_params - expected_params
            if unexpected_params:
                warnings.append(f"Unexpected parameters: {list(unexpected_params)}")
            
            # Check parameter types
            for param_name, param_value in args.items():
                if param_name in tool_metadata.parameters:
                    param_info = tool_metadata.parameters[param_name]
                    expected_type = param_info.get('type')
                    
                    if expected_type and expected_type != Any:
                        # Basic type checking
                        if expected_type == str and not isinstance(param_value, str):
                            warnings.append(f"Parameter '{param_name}' should be string, got {type(param_value).__name__}")
                        elif expected_type == int and not isinstance(param_value, int):
                            warnings.append(f"Parameter '{param_name}' should be int, got {type(param_value).__name__}")
                        elif expected_type == bool and not isinstance(param_value, bool):
                            warnings.append(f"Parameter '{param_name}' should be bool, got {type(param_value).__name__}")
            
            return "; ".join(warnings) if warnings else None
            
        except Exception as e:
            logger.error(f"Error validating arguments for {tool_name}: {e}")
            return f"Validation error: {e}"

    async def _get_detailed_error_info(self, tool_name: str, args: Union[str, Dict[str, Any]], error: Exception) -> str:
        """
        Get detailed error information for debugging tool execution issues.
        """
        try:
            # Get tool signature information
            signature_info = self.auto_manager.get_tool_signature_info(tool_name)
            
            error_info = []
            error_info.append(f"Tool: {tool_name}")
            error_info.append(f"Error type: {type(error).__name__}")
            error_info.append(f"Error message: {str(error)}")
            
            if signature_info:
                error_info.append(f"Expected signature:")
                error_info.append(f"  Service: {signature_info['service']}")
                error_info.append(f"  Method: {signature_info['method']}")
                error_info.append(f"  Async: {signature_info['is_async']}")
                error_info.append(f"  Return type: {signature_info['return_type']}")
                error_info.append(f"  Parameters:")
                for param_name, param_info in signature_info['parameters'].items():
                    required = "required" if param_info['required'] else "optional"
                    error_info.append(f"    {param_name}: {param_info['type']} ({required})")
            
            error_info.append(f"Provided arguments: {args}")
            
            return "\n".join(error_info)
            
        except Exception as e:
            return f"Could not get detailed error info: {e}"

    async def _parse_and_convert_args(self, tool_name: str, args: str) -> Dict[str, Any]:
        """
        Parse and convert string arguments to structured format.
        
        This is the unified entry point that routes everything through the generic LLM-based conversion.
        """
        try:
            # Get the tool metadata from the auto manager
            tool_metadata = self.auto_manager.get_tool_metadata(tool_name)
            if not tool_metadata or not tool_metadata.parameters:
                logger.error(f"Tool '{tool_name}' not found or has no parameters for argument parsing.")
                return {"error": f"Tool '{tool_name}' not found."}
            
            expected_parameters = tool_metadata.parameters
            
            # ALL tools now use the generic LLM-based conversion
            # This ensures consistent, intelligent parsing across all tools
            logger.debug(f"Using generic LLM-based parsing for tool '{tool_name}'")
            return await convert_natural_language_to_structured_args(
                self.llm, tool_name, args, expected_parameters
            )
            
        except Exception as e:
            logger.error(f"Error parsing arguments for tool '{tool_name}': {e}")
            # Fallback to simple query format
            return {"query": args}

    async def _maybe_await(self, result):
        """Helper to await async results or return sync results."""
        if hasattr(result, '__await__'):
            return await result
        return result

    async def get_tool_confirmation_message(self, tool_name: str, args: Any) -> str:
        """Generate a confirmation message for tool execution."""
        try:
            prompt = f"""You are a helpful personal trainer AI assistant. The user has requested an action that requires using the {tool_name} tool.\n\nTool arguments: {args}\n\nPlease provide a simple, natural statement that:\n1. Clearly states what action will be taken\n2. Includes the key details from the arguments in a user-friendly format\n3. Is concise and context-appropriate\n4. Does NOT ask for confirmation or end with a question\n\nExample formats:\n- For calendar events: \"I'll schedule a [workout type] for [time] at [location]\"\n- For location searches: \"I'll search for [location type] near [location]\"\n- For task creation: \"I'll create a task to [task description] due [date]\"\n- For calendar clearing: \"I'll clear your calendar for [time period]\"\n- For preferences: \"I'll remember that you like [preference]\"\n\nPlease provide the action statement:"""
            messages = [
                SystemMessage(content="You are a helpful personal trainer AI assistant. Always respond in clear, natural language. Be concise and direct in stating what action you're about to take."),
                HumanMessage(content=prompt)
            ]
            
            # Add timeout to prevent hanging
            try:
                response = await asyncio.wait_for(
                    self.llm.ainvoke(messages),
                    timeout=10.0  # 10 second timeout
                )
            except asyncio.TimeoutError:
                logger.warning(f"LLM call timed out in get_tool_confirmation_message for tool: {tool_name}")
                return "I'm about to process your request."
            
            return response.content.strip() if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"Error generating tool confirmation message: {e}")
            return "I'm about to process your request."

    async def summarize_tool_result(self, tool_name: str, tool_result: Any) -> str:
        """Summarize the result of a tool execution."""
        try:
            # Special handling for delete_events_in_range
            if tool_name == "delete_events_in_range":
                if isinstance(tool_result, int):
                    if tool_result == 0:
                        return "I've checked your calendar for the specified time period, but there were no events to delete."
                    elif tool_result == 1:
                        return "I've removed 1 event from your calendar for the specified time period."
                    else:
                        return f"I've removed {tool_result} events from your calendar for the specified time period."
                else:
                    return "I've cleared your calendar for the specified time period."
            
            prompt = get_tool_result_summary_prompt(tool_name, json.dumps(tool_result, default=str))
            messages = [
                SystemMessage(content="You are a helpful personal trainer AI assistant. Always respond in clear, natural language, never as a code block or raw data. Be encouraging and focused on helping the user achieve their fitness goals."),
                HumanMessage(content=prompt)
            ]
            
            # Add timeout to prevent hanging
            try:
                response = await asyncio.wait_for(
                    self.llm.ainvoke(messages),
                    timeout=10.0  # 10 second timeout
                )
            except asyncio.TimeoutError:
                logger.warning(f"LLM call timed out in summarize_tool_result for tool: {tool_name}")
                return self._get_fallback_summary(tool_name, tool_result)
            
            if not response or not hasattr(response, 'content') or not response.content.strip():
                raise RuntimeError("LLM returned empty response")
            
            summary = response.content.strip()
            if json.dumps(tool_result, default=str) in summary:
                raise RuntimeError("LLM returned raw tool result instead of a summary")
            if tool_name == "get_calendar_events":
                event_titles = [event.get('summary', '') for event in tool_result if isinstance(event, dict)]
                if not any(title in summary for title in event_titles):
                    if not event_titles:
                        return "You have no upcoming events in the requested time frame."
                    events = []
                    for event in tool_result:
                        start = event.get('start', {}).get('dateTime', event.get('start', {}).get('date', ''))
                        summary_title = event.get('summary', 'Untitled Event')
                        events.append(f"- {summary_title} at {start}")
                    return "Here are your upcoming events in the requested time frame:\n" + "\n".join(events)
            return summary
        except Exception as e:
            logger.error(f"Error summarizing tool result: {e}")
            return self._get_fallback_summary(tool_name, tool_result)

    async def _resolve_calendar_conflict(self, conflict_data: Union[str, Dict[str, Any]]) -> str:
        """Resolve calendar conflicts using the calendar service."""
        result = await self.services['calendar'].resolve_conflict(conflict_data)
        
        # If result is a dictionary, convert to a user-friendly string
        if isinstance(result, dict):
            if result.get('error'):
                return f"Error resolving conflict: {result['error']}"
            elif result.get('skipped'):
                return result.get('message', 'Event creation skipped due to conflict.')
            elif result.get('id'):  # Successfully created event
                return f"Conflict resolved successfully. Event created: {result.get('summary', 'Untitled Event')}"
            else:
                return f"Conflict resolution completed: {result.get('message', 'Unknown result')}"
        
        # If it's already a string, return as-is
        return str(result)

    def _get_fallback_summary(self, tool_name: str, tool_result: Any) -> str:
        """Generate a fallback summary when LLM summarization fails."""
        if tool_name == "get_calendar_events":
            if not tool_result or len(tool_result) == 0:
                return "You have no upcoming events in the requested time frame."
            else:
                return f"Found {len(tool_result)} upcoming events."
        elif tool_name == "create_calendar_event":
            return "Calendar event created successfully."
        elif tool_name == "send_email":
            return "Email sent successfully."
        elif tool_name == "create_task":
            return "Task created successfully."
        elif tool_name == "get_tasks":
            if not tool_result or len(tool_result) == 0:
                return "You have no pending tasks."
            else:
                return f"Found {len(tool_result)} pending tasks."
        else:
            return f"Successfully executed {tool_name}." 