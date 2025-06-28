"""
Agent State Machine for Personal Trainer AI

This module contains the state machine logic for orchestrating agent conversations,
including decision-making, tool execution, and response generation.

STATE MACHINE ARCHITECTURE:
   - `AgentStateMachine` - Uses AgentState as single source of truth
   - `AgentTransitionMachine` - Inherits from base, adds transition validation
   - Uses `AgentState` enum for type safety
   - `StateHandler` classes for each state (imported from state_handler.py)
   - `StateTransitionGraph` for transition validation
   - AgentState object for all conversation data

Example usage:
```python
# Basic state machine (no transition validation)
state_machine = AgentStateMachine(llm, tools, extract_pref, extract_time)

# Advanced state machine with transition validation
transition_machine = AgentTransitionMachine(llm, tools, extract_pref, extract_time)
```

"""

import json
import logging
from typing import List, Dict, Any, Optional, Union, AsyncGenerator, Callable
from datetime import datetime, timezone as dt_timezone
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.tools import Tool
from backend.utilities.time_formatting import extract_timeframe_from_text
from backend.agent_orchestration.utilities import convert_natural_language_to_structured_args
import asyncio
import dateparser
import inspect

# Import state handling functionality from separate module
from .state_handler import (
    AgentState,
    StateHandler,
    ThinkingStateHandler,
    ConfirmationStateHandler,
    ToolCallStateHandler,
    SummarizeToolResultStateHandler,
    ErrorStateHandler,
    StateTransitionGraph
)
from .agent_state import AgentState as AgentStateData

logger = logging.getLogger(__name__)


class AgentStateMachine:
    """
    State machine for orchestrating agent conversations and tool execution.
    
    This state machine uses AgentState as the single source of truth for all
    conversation data. It focuses on state transitions while delegating data
    management to the AgentState object.
    
    The state machine follows this flow:
    1. AGENT_THINKING - Analyze input and decide next action
    2. AGENT_CONFIRMATION - Confirm tool execution (if needed)
    3. AGENT_TOOL_CALL - Execute tool if needed
    4. AGENT_SUMMARIZE_TOOL_RESULT - Summarize tool results
    5. DONE - Complete the conversation turn
    """
    
    def __init__(self, llm, tools: List[Tool], extract_preference_func, extract_timeframe_func):
        """
        Initialize the state machine.
        
        Args:
            llm: The language model for decision making and summarization
            tools: List of available tools
            extract_preference_func: Function to extract user preferences
            extract_timeframe_func: Function to extract timeframes from text
        """
        self.llm = llm
        self.tools = tools
        self.extract_preference_func = extract_preference_func
        self.extract_timeframe_func = extract_timeframe_func
        self.current_state = AgentState.THINKING  # Initialize with default state
        
        # Initialize state handlers
        self.state_handlers = {
            AgentState.THINKING: ThinkingStateHandler(self),
            AgentState.CONFIRMATION: ConfirmationStateHandler(),
            AgentState.TOOL_CALL: ToolCallStateHandler(),
            AgentState.SUMMARIZE_TOOL_RESULT: SummarizeToolResultStateHandler(),
            AgentState.ERROR: ErrorStateHandler(),
        }
    
    async def process_messages_stream(self, messages: List[BaseMessage], 
                                    execute_tool_func, 
                                    get_tool_confirmation_func,
                                    summarize_tool_result_func,
                                    agent_state: AgentStateData) -> AsyncGenerator[str, None]:
        """
        Process messages and return a streaming response with multi-step tool execution.
        
        This method uses the provided AgentState object as the single source of truth
        for all conversation data. It focuses on state transitions while the AgentState
        handles all data management.
        
        Args:
            messages: List of messages to process
            execute_tool_func: Function to execute tools
            get_tool_confirmation_func: Function to get tool confirmation messages
            summarize_tool_result_func: Function to summarize tool results
            agent_state: AgentState object containing all conversation data
            
        Yields:
            str: Response messages
        """
        try:
            # Update agent state with new messages
            await agent_state.update(messages=messages, status="active")
            
            # Get the last user message from agent state
            if not agent_state.messages:
                yield "I didn't receive any valid messages to process."
                return
            
            last_message = agent_state.messages[-1]
            if not isinstance(last_message, HumanMessage):
                yield "I need a user message to process."
                return
            
            # Initialize minimal context for state handlers
            context = {
                'agent_state': agent_state,
                'execute_tool_func': execute_tool_func,
                'get_tool_confirmation_func': get_tool_confirmation_func,
                'summarize_tool_result_func': summarize_tool_result_func,
            }
            
            # Start with thinking state
            current_state = AgentState.THINKING
            
            # State machine loop
            while current_state != AgentState.DONE:
                handler = self.state_handlers.get(current_state)
                if not handler:
                    logger.error(f"No handler found for state: {current_state}")
                    yield f"Error: Unknown state {current_state}"
                    current_state = AgentState.ERROR
                    continue
                
                try:
                    next_state, response = await handler.handle(context)
                    if response:
                        yield response
                    
                    # Special handling for confirmation state - immediately proceed to tool execution
                    if current_state == AgentState.CONFIRMATION and next_state == AgentState.TOOL_CALL:
                        # The confirmation message was sent, now execute the tool immediately
                        tool_handler = self.state_handlers.get(AgentState.TOOL_CALL)
                        if tool_handler:
                            tool_state, tool_message = await tool_handler.handle(context)
                            if tool_message:
                                yield tool_message
                            
                            # If tool execution was successful, proceed to summarization
                            if tool_state == AgentState.SUMMARIZE_TOOL_RESULT:
                                summary_handler = self.state_handlers.get(AgentState.SUMMARIZE_TOOL_RESULT)
                                if summary_handler:
                                    final_state, summary_message = await summary_handler.handle(context)
                                    if summary_message:
                                        yield summary_message
                                    current_state = final_state
                                    break
                            else:
                                current_state = tool_state
                        else:
                            current_state = next_state
                    else:
                        current_state = next_state
                        
                except Exception as e:
                    logger.error(f"Error in state {current_state}: {e}")
                    yield f"Sorry, something went wrong: {str(e)}"
                    current_state = AgentState.ERROR
                    
        except Exception as e:
            logger.error(f"Error in process_messages_stream: {e}")
            yield f"Sorry, something went wrong: {str(e)}"
            await agent_state.update(status="error")

    async def decide_next_action(self, agent_state: AgentStateData) -> Dict[str, Any]:
        """
        Decide the next action based on the agent state.
        
        This method uses the AgentState object to get conversation history
        instead of recontextualizing the data.
        
        Args:
            agent_state: AgentState object containing conversation data
            
        Returns:
            Dict containing action type and details
        """
        try:
            # Get conversation history from agent state
            if not agent_state.messages:
                return {"type": "message", "content": "I don't have any conversation history to work with."}
            
            # Get the last user message
            last_message = agent_state.messages[-1]
            if not isinstance(last_message, HumanMessage):
                return {"type": "message", "content": "I need a user message to process."}
            
            user_input = last_message.content
            
            # Create conversation history for LLM
            conversation_history = []
            for msg in agent_state.messages:
                if isinstance(msg, HumanMessage):
                    conversation_history.append(f"User: {msg.content}")
                elif isinstance(msg, AIMessage):
                    conversation_history.append(f"Assistant: {msg.content}")
            
            # Build the prompt for decision making
            tools_description = "\n".join([f"- {tool.name}: {tool.description}" for tool in self.tools])
            
            prompt = f"""You are a helpful personal trainer AI assistant. You have access to the following tools:

{tools_description}

Current time: {datetime.now().strftime('%I:%M %p')}
Current date: {datetime.now().strftime('%A, %B %d, %Y')}

IMPORTANT RULES:
1. ONLY use tools when explicitly needed for the user's request
2. For calendar events:
   - ONLY use create_calendar_event when the user explicitly wants to schedule something
   - ONLY use get_calendar_events when the user asks to see their schedule
   - ONLY use delete_events_in_range when the user wants to clear their calendar
   - If the user asks to see their schedule, list events only for the requested time frame (e.g., 'this week', 'next week', 'today'). Do NOT schedule a new event unless explicitly requested.
3. For emails:
   - ONLY use send_email when the user wants to send a message
4. For tasks:
   - ONLY use create_task when the user wants to create a task
5. For location searches:
   - ONLY use search_location when the user wants to find a place
6. For sheets:
   - ONLY use create_workout_tracker when the user wants to create a new workout tracking spreadsheet
   - ONLY use add_workout_entry when the user wants to log a workout
   - ONLY use add_nutrition_entry when the user wants to log nutrition information
   - ONLY use get_sheet_data when the user wants to view sheet data

When using tools:
1. For calendar events:
   - Use create_calendar_event with natural language description (e.g., "schedule a workout for tomorrow at 7pm")
   - Use get_calendar_events with timeframe (e.g., "tomorrow", "this week")
   - Use delete_events_in_range with start_time|end_time format
2. For emails:
   - Use send_email with natural language description
3. For tasks:
   - Use create_task with natural language description
4. For location searches:
   - Use search_location with location|query format
   - Use find_nearby_workout_locations with location|radius format
     Example: find_nearby_workout_locations: "One Infinite Loop, Cupertino, CA 95014|30"
5. For sheets:
   - Use create_workout_tracker with title format
   - Use add_workout_entry with natural language description
   - Use add_nutrition_entry with natural language description
   - Use get_sheet_data with spreadsheet_id|range_name format

Example tool calls:
- create_calendar_event: "schedule a weightlifting workout for tomorrow at 7pm"
- get_calendar_events: "tomorrow"
- delete_events_in_range: "2025-06-18T00:00:00-07:00|2025-06-18T23:59:59-07:00"
- send_email: "send a progress report to my coach"
- create_task: "track protein intake due Friday"
- search_location: "San Francisco|gym"
- create_workout_tracker: "My Workout Tracker"
- add_workout_entry: "log today's upper body workout"
- add_nutrition_entry: "log lunch with 500 calories"
- get_sheet_data: "spreadsheet_id|Workouts!A1:E10"

IMPORTANT: Only use tools when explicitly needed for the user's request. Do not make unnecessary tool calls.

When the user asks to schedule a workout:
1. ALWAYS use create_calendar_event with natural language description
2. The tool will automatically convert it to proper JSON format
3. Example: create_calendar_event: "schedule a weightlifting workout for tomorrow at 7pm at the gym"

When the user shares a preference:
1. Use add_preference_to_kg to remember it
2. Example: add_preference_to_kg: "weightlifting"

When the user wants to see their schedule:
1. Use get_calendar_events with the timeframe
2. Example: get_calendar_events: "tomorrow" or get_calendar_events: "this week"

Conversation history:
{chr(10).join(conversation_history)}

What should I do next? Respond with either:
RESPONSE: <your response message>
or
TOOL: <tool_name>
ARGS: <tool_arguments>"""

            # Make LLM call with timeout
            try:
                response = await asyncio.wait_for(
                    self.llm.ainvoke([SystemMessage(content=prompt)]),
                    timeout=15.0
                )
                response_text = response.content.strip()
            except asyncio.TimeoutError:
                logger.error(f"LLM call timed out in decide_next_action for input: {user_input}")
                return {"type": "message", "content": "I'm having trouble processing your request right now. Please try again in a moment."}
            
            # Parse the response
            if response_text.startswith("RESPONSE:"):
                content = response_text[9:].strip()
                return {"type": "message", "content": content}
            elif response_text.startswith("TOOL:"):
                # Extract tool name and args
                lines = response_text.split('\n')
                tool_line = lines[0]
                args_line = lines[1] if len(lines) > 1 else ""
                
                tool_name = tool_line[5:].strip()
                tool_args = args_line[5:].strip() if args_line.startswith("ARGS:") else ""
                
                # Validate tool exists
                tool_names = [tool.name for tool in self.tools]
                if tool_name not in tool_names:
                    return {"type": "message", "content": f"I don't have access to the '{tool_name}' tool. Available tools: {', '.join(tool_names)}"}
                
                return {
                    "type": "tool_call",
                    "tool": tool_name,
                    "args": tool_args
                }
            else:
                # Fallback: treat as a message response
                return {"type": "message", "content": response_text}
                
        except Exception as e:
            logger.error(f"Error in decide_next_action: {e}")
            return {"type": "message", "content": f"Sorry, something went wrong while deciding next action: {str(e)}"}

    async def _validate_and_format_tool_call(self, tool_name: str, tool_args: Any, user_input: str) -> Dict[str, Any]:
        """
        Validate and format a tool call for execution.
        
        This method handles the conversion of tool arguments to the proper format
        expected by the tool function. It uses the convert_natural_language_to_structured_args
        utility when needed for complex parameter conversion.
        
        Args:
            tool_name: Name of the tool to call
            tool_args: Arguments for the tool (can be string, dict, or other types)
            user_input: Original user input (for context)
            
        Returns:
            Dict with tool call information or None if validation fails
        """
        try:
            # Find the tool
            tool = None
            for t in self.tools:
                if t.name == tool_name:
                    tool = t
                    break
            
            if not tool:
                logger.error(f"Tool '{tool_name}' not found")
                return None
            
            # Get tool function signature
            sig = inspect.signature(tool.func)
            params = list(sig.parameters.keys())
            
            # Parse arguments
            parsed_args = tool_args
            
            # Use the utility function for complex parameter conversion
            try:
                # Get expected parameters for the tool
                expected_parameters = {}
                for param_name in params:
                    param = sig.parameters[param_name]
                    expected_parameters[param_name] = {
                        'type': param.annotation if param.annotation != inspect.Parameter.empty else Any,
                        'required': param.default == inspect.Parameter.empty,
                        'default': param.default if param.default != inspect.Parameter.empty else None
                    }
                
                parsed_args = await convert_natural_language_to_structured_args(
                    self.llm, tool_name, tool_args, expected_parameters
                )
            except Exception as e:
                logger.error(f"Error converting natural language to structured args: {e}")
                # Fall back to original logic
            
            # Original logic for simple cases
            if len(params) == 1:
                param_name = params[0]
                param = sig.parameters[param_name]
                # If the param expects a dict, pass as dict if possible
                if param.annotation in [dict, Dict, Any] or param.annotation == inspect.Parameter.empty:
                    if isinstance(parsed_args, dict):
                        args = {param_name: parsed_args}
                    else:
                        args = {param_name: parsed_args}
                else:
                    args = {param_name: parsed_args}
            elif len(params) > 1:
                # If parsed_args is a dict, map keys to params
                if isinstance(parsed_args, dict):
                    args = {k: parsed_args.get(k, None) for k in params}
                else:
                    # Fallback: assign the string to the first param, rest get None
                    args = {params[0]: parsed_args}
                    for k in params[1:]:
                        args[k] = None
            else:
                args = {}
            
            return {
                "type": "tool_call",
                "tool": tool_name,
                "args": args
            }
        except Exception as e:
            logger.error(f"Error validating tool call: {e}")
            return None

    def _convert_message(self, msg) -> Optional[BaseMessage]:
        """
        Convert a message to LangChain format.
        
        Args:
            msg: Message to convert
            
        Returns:
            Converted message or None if conversion fails
        """
        try:
            if isinstance(msg, BaseMessage):
                return msg
            elif isinstance(msg, dict):
                role = msg.get('role', '')
                content = msg.get('content', '')
                if role == 'user':
                    return HumanMessage(content=content)
                elif role == 'assistant':
                    return AIMessage(content=content)
                elif role == 'system':
                    return SystemMessage(content=content)
                else:
                    return BaseMessage(content=content)
            elif isinstance(msg, str):
                return HumanMessage(content=msg)
            else:
                logger.warning(f"Unknown message format: {type(msg)}")
                return None
        except Exception as e:
            logger.error(f"Error converting message: {e}")
            return None

    def _determine_event(self, current_state: AgentState, next_state: AgentState, context: Dict[str, Any]) -> str:
        """
        Determine the event that caused the state transition.
        
        Args:
            current_state: Current state
            next_state: Next state
            context: Current context
            
        Returns:
            Event string
        """
        if current_state == AgentState.THINKING:
            if next_state == AgentState.DONE:
                return 'message_response'
            elif next_state == AgentState.CONFIRMATION:
                return 'tool_call'
            else:
                return 'error'
        elif current_state == AgentState.CONFIRMATION:
            # Check user input for confirmation or cancellation
            agent_state = context.get('agent_state')
            if agent_state and agent_state.messages:
                user_input = agent_state.messages[-1].content.lower()
                if 'yes' in user_input or 'confirm' in user_input or 'sure' in user_input:
                    return 'confirmed'
                elif 'no' in user_input or 'cancel' in user_input:
                    return 'cancelled'
            return 'error'
        elif current_state == AgentState.TOOL_CALL:
            if next_state == AgentState.SUMMARIZE_TOOL_RESULT:
                return 'success'
            else:
                return 'error'
        elif current_state == AgentState.SUMMARIZE_TOOL_RESULT:
            if next_state == AgentState.DONE:
                return 'success'
            else:
                return 'error'
        else:
            return 'error'


class AgentTransitionMachine(AgentStateMachine):
    """
    Advanced state machine with transition graph validation.
    Inherits from AgentStateMachine and adds transition validation capabilities.
    """
    
    def __init__(self, llm, tools: List[Tool], extract_preference_func, extract_timeframe_func):
        """
        Initialize the transition machine with transition graph support.
        
        Args:
            llm: The language model for decision making and summarization
            tools: List of available tools
            extract_preference_func: Function to extract user preferences
            extract_timeframe_func: Function to extract timeframes from text
        """
        # Call parent constructor
        super().__init__(llm, tools, extract_preference_func, extract_timeframe_func)
        
        # Initialize transition graph
        self.transition_graph = StateTransitionGraph()
    
    async def process_messages_stream(self, messages: List[BaseMessage], 
                                    execute_tool_func, 
                                    get_tool_confirmation_func,
                                    summarize_tool_result_func,
                                    agent_state: AgentStateData) -> AsyncGenerator[str, None]:
        """
        Process messages using the advanced state machine with transition graph validation.
        
        Args:
            messages: List of messages to process
            execute_tool_func: Function to execute tools
            get_tool_confirmation_func: Function to get tool confirmation messages
            summarize_tool_result_func: Function to summarize tool results
            agent_state: AgentState object containing all conversation data
            
        Yields:
            str: Response messages
        """
        try:
            # Update agent state with new messages
            await agent_state.update(messages=messages, status="active")
            
            # Get the last user message from agent state
            if not agent_state.messages:
                yield "I didn't receive any valid messages to process."
                return
            
            last_message = agent_state.messages[-1]
            if not isinstance(last_message, HumanMessage):
                yield "I need a user message to process."
                return
            
            # Initialize minimal context for state handlers
            context = {
                'agent_state': agent_state,
                'execute_tool_func': execute_tool_func,
                'get_tool_confirmation_func': get_tool_confirmation_func,
                'summarize_tool_result_func': summarize_tool_result_func,
            }
            
            # Start with thinking state
            current_state = AgentState.THINKING
            
            # State machine loop with transition graph validation
            while current_state != AgentState.DONE:
                # Get handler for current state
                handler = self.state_handlers.get(current_state)
                if not handler:
                    logger.error(f"No handler found for state: {current_state}")
                    yield f"Error: Unknown state {current_state}"
                    break
                
                # Handle current state
                try:
                    next_state, response = await handler.handle(context)
                    
                    # Yield response if any
                    if response:
                        yield response
                    
                    # Use transition graph to validate transition
                    event = self._determine_event(current_state, next_state, context)
                    validated_next_state = self.transition_graph.get_next_state(current_state, event)
                    
                    if validated_next_state != next_state:
                        logger.warning(f"Invalid transition from {current_state} to {next_state} via {event}")
                        # Fall back to validated transition or DONE
                        next_state = validated_next_state or AgentState.DONE
                    
                    # Transition to next state
                    current_state = next_state
                    
                except Exception as e:
                    logger.error(f"Error in state {current_state}: {e}")
                    yield f"Error processing state {current_state}: {str(e)}"
                    # Transition to DONE on error
                    current_state = AgentState.DONE
                
        except Exception as e:
            logger.error(f"Error in process_messages_stream: {e}")
            yield f"Error processing messages: {str(e)}"
            await agent_state.update(status="error") 