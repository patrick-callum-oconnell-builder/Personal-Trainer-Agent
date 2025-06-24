"""
utilities.py

Utility functions for agent orchestration, including natural language processing
and tool argument conversion.
"""

import json
import logging
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


async def convert_natural_language_to_structured_args(
    llm: ChatOpenAI,
    tool_name: str, 
    natural_language_input: str, 
    expected_parameters: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Convert natural language input to structured arguments using LLM.
    
    This is a generic approach that works for any tool.
    
    Args:
        llm: The language model to use for conversion
        tool_name: Name of the tool being called
        natural_language_input: User's natural language input
        expected_parameters: Dictionary describing the tool's expected parameters
        
    Returns:
        Dict containing the structured arguments for the tool
    """
    try:
        # Get current date context for calendar events
        current_date = datetime.now(timezone.utc)
        date_context = f"Current date: {current_date.strftime('%Y-%m-%d')} (UTC)"
        
        # Create a prompt that describes the tool and its expected parameters
        param_descriptions = []
        for param_name, param_info in expected_parameters.items():
            param_type = param_info.get('type', 'any')
            required = param_info.get('required', True)
            default = param_info.get('default', None)
            
            # Get a more readable type name
            if hasattr(param_type, '__name__'):
                type_name = param_type.__name__
            elif hasattr(param_type, '__origin__'):
                type_name = str(param_type)
            else:
                type_name = str(param_type)
            
            desc = f"- {param_name} ({type_name})"
            if not required:
                desc += f" (optional, default: {default})"
            
            # Add special guidance for complex types and specific tools
            if type_name == 'Dict' or 'Dict' in str(param_type):
                if param_name == 'event_details':
                    desc += " - Should be a JSON object with 'summary', 'start', and 'end' fields for calendar events"
                else:
                    desc += " - Should be a JSON object"
            elif type_name == 'List' or 'List' in str(param_type):
                desc += " - Should be a JSON array"
            
            param_descriptions.append(desc)
        
        # Add tool-specific guidance
        tool_guidance = _get_tool_specific_guidance(tool_name)
        
        # Add tool-specific examples
        tool_examples = _get_tool_examples(tool_name)
        examples_text = ""
        if tool_examples:
            examples_text = f"\n\nExample inputs for this tool:\n" + "\n".join([f"- {ex}" for ex in tool_examples[:3]])
        
        # Add date context for calendar events
        date_guidance = ""
        if tool_name == "create_calendar_event":
            date_guidance = f"\n\nIMPORTANT: {date_context}\nWhen parsing dates like 'tomorrow', 'next week', etc., use the current date as reference.\nFor example, if today is {current_date.strftime('%Y-%m-%d')}, then 'tomorrow' would be {(current_date + timedelta(days=1)).strftime('%Y-%m-%d')}."
        
        prompt = f"""Convert this natural language input into structured arguments for the {tool_name} tool.

Tool: {tool_name}
Expected parameters:
{chr(10).join(param_descriptions)}
{tool_guidance}
{examples_text}
{date_guidance}

Natural language input: "{natural_language_input}"

Respond ONLY with a valid JSON object containing the parameter values. Do not include any explanation or text outside the JSON.

Example format:
{{
    "param1": "value1",
    "param2": "value2"
}}

JSON response:"""

        messages = [
            SystemMessage(content="You are a helpful AI assistant that converts natural language to structured tool arguments. Always respond with valid JSON only."),
            HumanMessage(content=prompt)
        ]
        
        # Add timeout to prevent hanging
        try:
            response = await asyncio.wait_for(
                llm.ainvoke(messages),
                timeout=10.0  # 10 second timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"LLM call timed out for {tool_name}, using simple fallback")
            return {"query": natural_language_input}
        
        if not response or not hasattr(response, 'content') or not response.content.strip():
            logger.warning(f"LLM returned empty response for {tool_name}, using simple fallback")
            return {"query": natural_language_input}
        
        # Extract JSON from response
        content = response.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        # Validate JSON
        try:
            parsed_args = json.loads(content)
            
            # Validate that all required parameters are present
            for param_name, param_info in expected_parameters.items():
                if param_info.get('required', True) and param_name not in parsed_args:
                    # Try to provide a reasonable default
                    if param_info.get('type') == str:
                        parsed_args[param_name] = ""
                    elif param_info.get('type') == int:
                        parsed_args[param_name] = 0
                    elif param_info.get('type') == bool:
                        parsed_args[param_name] = False
                    elif hasattr(param_info.get('type'), '__origin__') and param_info.get('type').__origin__ is list:
                        parsed_args[param_name] = []
                    elif hasattr(param_info.get('type'), '__origin__') and param_info.get('type').__origin__ is dict:
                        parsed_args[param_name] = {}
                    else:
                        parsed_args[param_name] = None
            
            return parsed_args
            
        except json.JSONDecodeError:
            logger.warning(f"LLM returned invalid JSON for {tool_name}, using simple fallback")
            return {"query": natural_language_input}
            
    except Exception as e:
        logger.error(f"Error converting natural language to structured args for {tool_name}: {e}")
        # Simple fallback
        return {"query": natural_language_input}


def _get_tool_specific_guidance(tool_name: str) -> str:
    """Get tool-specific guidance for the LLM prompt."""
    guidance_map = {
        "create_calendar_event": """
        For calendar events, extract the event details directly (not wrapped in a parameter):
        - summary: The event title/description
        - start: Start time in ISO format (YYYY-MM-DDTHH:MM:SS)
        - end: End time in ISO format (YYYY-MM-DDTHH:MM:SS)
        - description: Optional detailed description
        - location: Optional location
        
        Example: "schedule a workout for tomorrow at 3pm" should become:
        {
            "summary": "Workout",
            "start": "2024-01-24T15:00:00",
            "end": "2024-01-24T16:00:00",
            "description": "Scheduled workout session"
        }
        
        Note: Do NOT wrap in an "event_details" parameter. Generate the event data directly.
        """,
        
        "send_email": """
        For emails, extract:
        - recipient: Email address or name
        - subject: Email subject line
        - body: Email content
        
        Example: "send an email to john@example.com about the meeting tomorrow" should become:
        {
            "recipient": "john@example.com",
            "subject": "Meeting Tomorrow",
            "body": "Hi John, I wanted to discuss the meeting scheduled for tomorrow."
        }
        """,
        
        "create_task": """
        For tasks, extract:
        - title: Task title
        - description: Task description
        - due_date: Optional due date in ISO format
        
        Example: "create a task to buy groceries by Friday" should become:
        {
            "title": "Buy groceries",
            "description": "Purchase groceries for the week",
            "due_date": "2024-01-26T23:59:59"
        }
        """,
        
        "get_nearby_locations": """
        For location searches, extract:
        - query: What to search for (e.g., "gym", "restaurant")
        - location: Optional specific location to search around
        
        Example: "find gyms near downtown" should become:
        {
            "query": "gym",
            "location": "downtown"
        }
        """,
        
        "get_directions": """
        For directions, extract:
        - origin: Starting location
        - destination: End location
        - mode: Optional travel mode (driving, walking, transit)
        
        Example: "get directions from home to the gym" should become:
        {
            "origin": "home",
            "destination": "gym",
            "mode": "driving"
        }
        """
    }
    
    return guidance_map.get(tool_name, "")


def _get_tool_examples(tool_name: str) -> List[str]:
    """Get example inputs for a specific tool."""
    try:
        # Import tool configuration
        from backend.tools.tool_config import TOOL_METADATA
        
        # Search through all services for the tool
        for service_name, service_tools in TOOL_METADATA.items():
            for method_name, tool_info in service_tools.items():
                if tool_info['name'] == tool_name:
                    return tool_info.get('examples', [])
        
        return []
    except Exception as e:
        logger.error(f"Error getting tool examples for {tool_name}: {e}")
        return [] 