"""
Tools module for the Personal Trainer AI application.

This module contains all tool-related functionality including:
- Tool definitions and creation
- Tool execution logic
- Tool result processing and summarization
- Tool argument parsing and validation
- Calendar-specific tool operations
"""

import json
import logging
import asyncio
from datetime import datetime, timedelta, timezone as dt_timezone
from typing import List, Dict, Any, Optional, Union
from pydantic.v1 import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

from backend.google_services import (
    GoogleCalendarService,
    GoogleDriveService,
    GoogleFitnessService,
    GoogleGmailService,
    GoogleMapsService,
    GoogleSheetsService,
    GoogleTasksService,
)
from backend.tools import (
    get_calendar_events,
    add_preference_to_kg,
)
from backend.time_formatting import extract_timeframe_from_text
from backend.tools.maps_tools import FindNearbyWorkoutLocationsInput

logger = logging.getLogger(__name__)


class ToolManager:
    """
    Manages all tool-related operations for the Personal Trainer AI.
    
    This class encapsulates all tool creation, execution, and processing logic,
    providing a clean interface for the agent to interact with tools.
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
        """Initialize the tool manager with Google services."""
        self.calendar_service = calendar_service
        self.gmail_service = gmail_service
        self.tasks_service = tasks_service
        self.drive_service = drive_service
        self.sheets_service = sheets_service
        self.maps_service = maps_service
        self.llm = llm or ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.7,
            streaming=True
        )
        self.tools: List[Tool] = []
        self._create_tools()
    
    def _create_tools(self) -> None:
        """Create and register all available tools."""
        self.tools = []
        
        # Calendar tools
        if self.calendar_service:
            logger.debug("Adding Calendar tools...")
            self.tools.extend([
                Tool(
                    name="get_calendar_events",
                    func=self.calendar_service.get_upcoming_events,
                    description="Get upcoming calendar events"
                ),
                Tool(
                    name="create_calendar_event",
                    func=self.calendar_service.write_event,
                    description="Create a new calendar event"
                ),
                Tool(
                    name="resolve_calendar_conflict",
                    func=self._resolve_calendar_conflict,
                    description="Resolve calendar conflicts by replacing, deleting, or skipping conflicting events"
                ),
                Tool(
                    name="delete_events_in_range",
                    func=self.calendar_service.delete_events_in_range,
                    description="Delete all calendar events within a specified time range"
                )
            ])
        
        # Gmail tools
        if self.gmail_service:
            logger.debug("Adding Gmail tools...")
            self.tools.extend([
                Tool(
                    name="send_email",
                    func=self.gmail_service.send_message,
                    description="Send an email to a recipient"
                )
            ])
        
        # Tasks tools
        if self.tasks_service:
            logger.debug("Adding Tasks tools...")
            self.tools.extend([
                Tool(
                    name="create_task",
                    func=self.tasks_service.create_task,
                    description="Create a new task"
                ),
                Tool(
                    name="get_tasks",
                    func=self.tasks_service.get_tasks,
                    description="Get tasks from the task list"
                )
            ])
        
        # Drive tools
        if self.drive_service:
            logger.debug("Adding Drive tools...")
            self.tools.extend([
                Tool(
                    name="search_drive",
                    func=self.drive_service.search_files,
                    description="Search for files in Google Drive"
                )
            ])
        
        # Sheets tools
        if self.sheets_service:
            logger.debug("Adding Sheets tools...")
            self.tools.extend([
                Tool(
                    name="get_sheet_data",
                    func=self.sheets_service.get_sheet_data,
                    description="Get data from a Google Sheet"
                ),
                Tool(
                    name="create_workout_tracker",
                    func=self.sheets_service.create_workout_tracker,
                    description="Create a new workout tracking spreadsheet"
                ),
                Tool(
                    name="add_workout_entry",
                    func=self.sheets_service.add_workout_entry,
                    description="Add a workout entry to the tracker"
                ),
                Tool(
                    name="add_nutrition_entry",
                    func=self.sheets_service.add_nutrition_entry,
                    description="Add a nutrition entry to the tracker"
                )
            ])
        
        # Maps tools
        if self.maps_service:
            logger.debug("Adding Maps tools...")
            self.tools.extend([
                Tool(
                    name="get_directions",
                    func=self.maps_service.get_directions,
                    description="Get directions between two locations"
                ),
                Tool(
                    name="find_nearby_workout_locations",
                    func=self.maps_service.find_nearby_workout_locations,
                    description="Find nearby workout locations (gyms, fitness centers, etc.) near a given location"
                )
            ])
        
        # Knowledge graph tools
        self.tools.append(
            Tool(
                name="add_preference_to_kg",
                func=add_preference_to_kg,
                description="Add a user preference to the knowledge graph"
            )
        )
        
        logger.info(f"Created {len(self.tools)} tools for agent")
    
    def get_tools(self) -> List[Tool]:
        """Get the list of available tools."""
        return self.tools
    
    def get_tool_by_name(self, tool_name: str) -> Optional[Tool]:
        """Get a specific tool by name."""
        return next((tool for tool in self.tools if tool.name == tool_name), None)
    
    async def execute_tool(self, tool_name: str, args: Union[str, Dict[str, Any]]) -> Any:
        """Execute a tool with the given arguments."""
        try:
            # Find the tool
            tool = self.get_tool_by_name(tool_name)
            if not tool:
                raise ValueError(f"Tool {tool_name} not found")

            # Special handling for add_preference_to_kg
            if tool_name == "add_preference_to_kg":
                # If args is a dict with 'query', extract the value
                if isinstance(args, dict) and 'query' in args:
                    arg_val = args['query']
                else:
                    arg_val = args
                result = await tool.func(arg_val) if asyncio.iscoroutinefunction(tool.func) else tool.func(arg_val)
                logger.info(f"Tool {tool_name} returned: {result}")
                return result

            # Convert string args to dict if needed
            if isinstance(args, str):
                args = self._parse_string_args(tool_name, args)

            # Execute the tool
            logger.info(f"Executing tool {tool_name} with args: {args}")
            result = await tool.func(args) if asyncio.iscoroutinefunction(tool.func) else tool.func(args)
            logger.info(f"Tool {tool_name} returned: {result}")
            return result

        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {str(e)}")
            raise
    
    def _parse_string_args(self, tool_name: str, args: str) -> Dict[str, Any]:
        """Parse string arguments into the appropriate format for each tool."""
        args = args.strip('"')
        
        # Special handling for get_calendar_events
        if tool_name == "get_calendar_events":
            # Extract time frame from the user's query
            timeframe = extract_timeframe_from_text(args)
            if timeframe:
                return timeframe
            else:
                # Default to upcoming events if no time frame specified
                return {"timeMin": datetime.now(dt_timezone.utc).isoformat()}
        
        # Special handling for delete_events_in_range
        elif tool_name == "delete_events_in_range":
            # Parse pipe-separated format: start_time|end_time
            parts = args.split("|")
            if len(parts) >= 2:
                return {
                    "start_time": parts[0],
                    "end_time": parts[1]
                }
            else:
                raise ValueError(f"Invalid time range format. Expected 'start_time|end_time', got: {args}")
        
        # Special handling for create_calendar_event
        elif tool_name == "create_calendar_event":
            # This will be handled by the calendar event converter
            return {"natural_language_input": args}
        
        # Special handling for send_email
        elif tool_name == "send_email":
            # Parse pipe-separated format: recipient|subject|body
            parts = args.split("|")
            if len(parts) >= 3:
                return {
                    "to": parts[0],
                    "subject": parts[1],
                    "body": parts[2]
                }
            else:
                raise ValueError(f"Invalid email format. Expected 'recipient|subject|body', got: {args}")
        
        # Special handling for find_nearby_workout_locations
        elif tool_name == "find_nearby_workout_locations":
            # Parse pipe-separated format: address|radius
            parts = args.split("|")
            if len(parts) == 2:
                address = parts[0].strip()
                try:
                    radius = int(parts[1].strip())
                except Exception:
                    radius = 30
                # Geocode the address to get lat/lng
                if hasattr(self.maps_service, 'geocode_address'):
                    # Note: This would need to be async, but we're in a sync context
                    # For now, return the location format
                    return {"location": address, "radius": radius}
                else:
                    return {"location": address, "radius": radius}
            else:
                return {"location": args}
        
        else:
            # For other tools, just pass the string as is
            return {"query": args} if args else {}
    
    async def _resolve_calendar_conflict(self, conflict_data: Union[str, Dict[str, Any]]) -> str:
        """Handle calendar conflict resolution."""
        try:
            if isinstance(conflict_data, str):
                conflict_data = json.loads(conflict_data)
            
            event_details = conflict_data.get("event_details")
            conflict_action = conflict_data.get("action", "skip")  # skip, replace, or delete
            
            if not event_details:
                return "Error: Missing event_details in conflict resolution request"
            
            # Convert to proper format if needed
            if isinstance(event_details, str):
                event_details = json.loads(event_details)
            
            event_details = self._convert_to_calendar_format(event_details)
            result = await self.calendar_service.write_event_with_conflict_resolution(event_details, conflict_action)
            
            if isinstance(result, dict) and result.get("type") == "conflict":
                return f"Conflict still exists after resolution attempt: {result['message']}"
            else:
                return f"Successfully created event with conflict resolution (action: {conflict_action})"
                
        except Exception as e:
            logger.error(f"Error resolving calendar conflict: {e}")
            return f"Error resolving conflict: {str(e)}"
    
    def _convert_to_calendar_format(self, event_details: Dict[str, Any]) -> Dict[str, Any]:
        """Convert event details to Google Calendar format."""
        converted = event_details.copy()
        
        # Get user timezone (default to Pacific Time)
        user_tz = pytz.timezone('America/Los_Angeles')
        
        # Convert start time if it's a string
        if 'start' in converted and isinstance(converted['start'], str):
            # Parse the datetime string and add timezone
            try:
                dt = datetime.fromisoformat(converted['start'].replace('Z', '+00:00'))
                if dt.tzinfo is None:
                    dt = user_tz.localize(dt)
                converted['start'] = {'dateTime': dt.isoformat()}
            except ValueError:
                # If parsing fails, try to add timezone to the string
                if not converted['start'].endswith('Z') and '+' not in converted['start']:
                    converted['start'] = {'dateTime': converted['start'] + '-08:00'}  # PST
                else:
                    converted['start'] = {'dateTime': converted['start']}
        
        # Convert end time if it's a string
        if 'end' in converted and isinstance(converted['end'], str):
            # Parse the datetime string and add timezone
            try:
                dt = datetime.fromisoformat(converted['end'].replace('Z', '+00:00'))
                if dt.tzinfo is None:
                    dt = user_tz.localize(dt)
                converted['end'] = {'dateTime': dt.isoformat()}
            except ValueError:
                # If parsing fails, try to add timezone to the string
                if not converted['end'].endswith('Z') and '+' not in converted['end']:
                    converted['end'] = {'dateTime': converted['end'] + '-08:00'}  # PST
                else:
                    converted['end'] = {'dateTime': converted['end']}
        
        return converted
    
    async def convert_natural_language_to_calendar_json(self, natural_language_input: str) -> str:
        """Convert natural language input to JSON format for calendar events using LLM."""
        pacific_tz = pytz.timezone('America/Los_Angeles')
        now = datetime.now(pacific_tz)
        
        def build_prompt(input_text: str) -> str:
            return f"""Convert this natural language event description into a Google Calendar event JSON.
Current time: {now.strftime('%Y-%m-%d %H:%M')} Pacific Time

Input: "{input_text}"

Respond ONLY with a valid JSON object, no text or explanation, and never repeat the input. The JSON must have these fields:
- summary: Event title
- start: Object with dateTime (ISO format with -07:00 timezone) and timeZone ("America/Los_Angeles")
- end: Object with dateTime (ISO format with -07:00 timezone) and timeZone ("America/Los_Angeles")
- description: Brief description (optional)
- location: Event location (optional)

Rules:
1. If no time is specified, use 6:00 PM tomorrow
2. If no duration is specified, make it 1 hour
3. Always use Pacific Time (-07:00)
4. For "tomorrow", use tomorrow's date
5. For "today", use today's date
6. For times like "9 AM", convert to 24-hour format (09:00)

Example:
{{
    "summary": "Workout Session",
    "start": {{
        "dateTime": "2024-03-20T18:00:00-07:00",
        "timeZone": "America/Los_Angeles"
    }},
    "end": {{
        "dateTime": "2024-03-20T19:00:00-07:00",
        "timeZone": "America/Los_Angeles"
    }},
    "description": "General fitness workout",
    "location": "Gym"
}}
"""
        
        last_json_string = None
        for attempt in range(2):  # Try twice
            try:
                messages = [
                    SystemMessage(content="You are a helpful assistant that converts natural language to Google Calendar event JSON. Always return valid JSON only. Never use hardcoded dates - always use relative dates based on the current date."),
                    HumanMessage(content=build_prompt(natural_language_input))
                ]
                response = await self.llm.ainvoke(messages)
                json_string = response.content.strip()
                json_string = json_string.replace('```json', '').replace('```', '').strip()
                last_json_string = json_string
                event_data = json.loads(json_string)
                
                # Ensure start and end are properly formatted
                for time_field in ['start', 'end']:
                    if time_field in event_data:
                        if isinstance(event_data[time_field], str):
                            dt = dateparser.parse(event_data[time_field], settings={'PREFER_DATES_FROM': 'future'})
                            if dt:
                                if dt.tzinfo is None:
                                    dt = pacific_tz.localize(dt)
                                event_data[time_field] = {
                                    'dateTime': dt.isoformat(),
                                    'timeZone': 'America/Los_Angeles'
                                }
                        elif isinstance(event_data[time_field], dict):
                            if 'dateTime' in event_data[time_field]:
                                dt = dateparser.parse(event_data[time_field]['dateTime'], settings={'PREFER_DATES_FROM': 'future'})
                                if dt:
                                    if dt.tzinfo is None:
                                        dt = pacific_tz.localize(dt)
                                    event_data[time_field]['dateTime'] = dt.isoformat()
                                    event_data[time_field]['timeZone'] = 'America/Los_Angeles'
                
                return json.dumps(event_data)
                
            except Exception as e:
                logger.error(f"Attempt {attempt+1}: Error converting natural language to JSON: {e}. LLM output: {last_json_string}")
                if attempt == 0:
                    continue
                else:
                    raise ValueError(f"LLM did not return valid JSON for event. LLM output: {last_json_string}")
    
    async def summarize_tool_result(self, tool_name: str, tool_result: Any) -> str:
        """Summarize a tool result using the LLM to provide a natural, user-friendly response."""
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

            # Create a detailed prompt for the LLM to summarize the tool result
            prompt = f"""You are a helpful personal trainer AI assistant. Summarize the result of the {tool_name} tool in a user-friendly way.

Tool result: {json.dumps(tool_result, default=str)}

Guidelines:
1. Be concise but informative
2. Use natural, conversational language
3. Format any dates, times, or numbers in a readable way
4. If there are any errors or issues, explain them clearly
5. If the result is a list or complex data, summarize the key points
6. Use markdown formatting for better readability
7. For calendar events, ALWAYS include:
   - Event title
   - Date and time in a readable format
   - A clickable link to the event using markdown [Event Link](url)
   - Any other relevant details
8. For workout locations, include:
   - Name of the location
   - Address
   - Distance if available
9. For tasks, include:
   - Task name
   - Due date
   - Priority if available
10. For emails, include:
    - Recipient
    - Subject
    - Status of the send operation

Example responses:
- For calendar events: "I've scheduled your Upper Body Workout for tomorrow at 10 AM at the Downtown Gym. You can view all the details here: [Event Link](https://calendar.google.com/event/...)"
- For workout locations: "I found a great gym nearby: Fitness First at 123 Main St, just 0.5 miles away. They have all the equipment you need for your workout routine."
- For tasks: "I've added 'Track daily protein intake' to your task list, due this Friday. I'll remind you about it as the deadline approaches."

Please provide a natural, detailed response:"""

            messages = [
                SystemMessage(content="You are a helpful personal trainer AI assistant. Always respond in clear, natural language, never as a code block or raw data. Be encouraging and focused on helping the user achieve their fitness goals."),
                HumanMessage(content=prompt)
            ]

            response = await self.llm.ainvoke(messages)
            if not response or not hasattr(response, 'content') or not response.content.strip():
                raise RuntimeError("LLM returned empty response")
            
            summary = response.content.strip()
            
            # Validate that the summary is not just a raw tool result
            if json.dumps(tool_result, default=str) in summary:
                raise RuntimeError("LLM returned raw tool result instead of a summary")
            
            # If get_calendar_events and summary does not mention any event titles, fall back to default event list
            if tool_name == "get_calendar_events":
                event_titles = [event.get('summary', '') for event in tool_result if isinstance(event, dict)]
                if not any(title in summary for title in event_titles):
                    # Fallback: list the events
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
    
    def _get_fallback_summary(self, tool_name: str, tool_result: Any) -> str:
        """Provide a basic fallback response based on the tool type."""
        if tool_name == "create_calendar_event":
            return "I've scheduled your workout in your calendar. You can check your calendar app for the details."
        elif tool_name == "get_calendar_events":
            return "I'll list your upcoming events."
        elif tool_name == "find_nearby_workout_locations":
            if not tool_result:
                return "I couldn't find any workout locations nearby."
            locations = []
            for location in tool_result:
                name = location.get('name', 'Unknown Location')
                address = location.get('address', 'No address available')
                rating = location.get('rating', 'No rating')
                locations.append(f"- {name} at {address} (Rating: {rating})")
            return "Here are some workout locations nearby:\n" + "\n".join(locations)
        elif tool_name == "delete_events_in_range":
            if isinstance(tool_result, int):
                return f"I've removed {tool_result} events from your calendar."
            return "I've cleared your calendar for the specified time period."
        else:
            return f"I've completed your request. You can check the details in your {tool_name.replace('_', ' ')}."
    
    async def get_tool_confirmation_message(self, tool_name: str, args: str) -> str:
        """Get a confirmation message for a tool call."""
        try:
            # Create a prompt that guides the LLM to generate a simple action statement
            prompt = f"""You are a helpful personal trainer AI assistant. The user has requested an action that requires using the {tool_name} tool.

Tool arguments: {args}

Please provide a simple, natural statement that:
1. Clearly states what action will be taken
2. Includes the key details from the arguments in a user-friendly format
3. Is concise and context-appropriate
4. Does NOT ask for confirmation or end with a question

Example formats:
- For calendar events: "I'll schedule a [workout type] for [time] at [location]"
- For location searches: "I'll search for [location type] near [location]"
- For task creation: "I'll create a task to [task description] due [date]"
- For calendar clearing: "I'll clear your calendar for [time period]"
- For preferences: "I'll remember that you like [preference]"

Please provide the action statement:"""

            messages = [
                SystemMessage(content="You are a helpful personal trainer AI assistant. Always respond in clear, natural language. Be concise and direct in stating what action you're about to take."),
                HumanMessage(content=prompt)
            ]

            response = await self.llm.ainvoke(messages)
            return response.content.strip() if hasattr(response, 'content') else str(response)

        except Exception as e:
            logger.error(f"Error generating tool confirmation message: {e}")
            return "I'm about to process your request." 