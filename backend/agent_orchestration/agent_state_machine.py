"""
Agent State Machine for Personal Trainer AI

This module contains the state machine logic for orchestrating agent conversations,
including decision-making, tool execution, and response generation.

STATE MACHINE ARCHITECTURE:
   - `AgentStateMachine` - Base class with all core functionality
   - `AgentTransitionMachine` - Inherits from base, adds transition validation
   - Uses `AgentState` enum for type safety
   - `StateHandler` classes for each state (imported from state_handler.py)
   - `StateTransitionGraph` for transition validation
   - Context dictionary for state data

   
Example usage:
```python
# Basic state machine (no transition validation)
state_machine = AgentStateMachine(llm, tools, extract_pref, extract_time)

# Advanced state machine with transition validation
transition_machine = AgentTransitionMachine(llm, tools, extract_pref, extract_time)
"""

import json
import logging
from typing import List, Dict, Any, Optional, Union, AsyncGenerator, Callable
from datetime import datetime, timezone as dt_timezone
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.tools import Tool
from backend.prompts import get_system_prompt
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
    ToolCallStateHandler,
    SummarizeToolResultStateHandler,
    ErrorStateHandler,
    StateTransitionGraph,
    ConfirmationStateHandler
)

logger = logging.getLogger(__name__)


class AgentStateMachine:
    """
    State machine for orchestrating agent conversations and tool execution.
    
    The state machine follows this flow:
    1. AGENT_THINKING - Analyze input and decide next action
    2. AGENT_TOOL_CALL - Execute tool if needed
    3. AGENT_SUMMARIZE_TOOL_RESULT - Summarize tool results
    4. DONE - Complete the conversation turn
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
                                    agent_state=None) -> AsyncGenerator[str, None]:
        """
        Process messages and return a streaming response with multi-step tool execution.
        Handles errors by transitioning to ERROR state and yielding user-facing error messages.
        """
        try:
            # Convert messages to LangChain format
            input_messages = []
            for msg in messages:
                converted = self._convert_message(msg)
                if converted:
                    input_messages.append(converted)
            
            if not input_messages:
                yield "I didn't receive any valid messages to process."
                return
            
            # Get the last user message
            user_message = input_messages[-1]
            if not isinstance(user_message, HumanMessage):
                yield "I need a user message to process."
                return
            
            # Initialize context
            context = {
                'history': [user_message.content],
                'user_input': user_message.content,
                'agent_state': agent_state,
                'execute_tool_func': execute_tool_func,
                'get_tool_confirmation_func': get_tool_confirmation_func,
                'summarize_tool_result_func': summarize_tool_result_func,
                'agent_action': None,
                'tool_result': None,
                'last_tool': None,
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
            current_state = AgentState.ERROR

    async def decide_next_action(self, history) -> Dict[str, Any]:
        """
        Decide the next action based on the conversation history.
        
        Args:
            history: Conversation history
            
        Returns:
            Dict containing action type and details
        """
        try:
            # Convert history to the format expected by the agent
            if isinstance(history, list):
                # Get the last message's content
                last_message = history[-1]
                if hasattr(last_message, 'content'):
                    input_text = last_message.content
                else:
                    input_text = str(last_message)
                # Get previous messages for chat history
                chat_history = []
                for msg in history[:-1]:
                    if hasattr(msg, 'content'):
                        chat_history.append(msg.content)
                    else:
                        chat_history.append(str(msg))
            else:
                input_text = str(history)
                chat_history = []
            
            # Get current time and date
            current_time = datetime.now().strftime("%I:%M %p")
            current_date = datetime.now().strftime("%A, %B %d, %Y")

            # Create the system prompt with tool descriptions
            system_prompt = get_system_prompt(self.tools, current_time, current_date)

            # Create the prompt with the full conversation history
            user_prompt = f"Conversation history:\n"
            for msg in chat_history:
                user_prompt += f"{msg}\n"
            user_prompt += f"\nUser's latest message: {input_text}\n\nWhat should I do next?"

            # Get the LLM's response using proper message objects
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            # Add timeout to prevent hanging
            try:
                response = await asyncio.wait_for(
                    self.llm.ainvoke(messages),
                    timeout=15.0  # 15 second timeout
                )
            except asyncio.TimeoutError:
                logger.error(f"LLM call timed out in decide_next_action for input: {input_text}")
                return {
                    "type": "message",
                    "content": "I'm having trouble processing your request right now. Please try again in a moment."
                }
            
            # Handle both string and AIMessage responses
            if isinstance(response, str):
                response_text = response
            elif hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
            
            if not response_text or not response_text.strip():
                logger.error(f"LLM returned empty response for input: {input_text}")
                raise RuntimeError("LLM returned empty response.")
            
            response_text = response_text.strip()
            
            # Try to extract tool call from the response
            tool_call = await self._extract_tool_call(response_text, input_text)
            if tool_call:
                return tool_call
            
            # Handle regular messages
            return {
                "type": "message",
                "content": response_text
            }
                
        except Exception as e:
            logger.error(f"Error deciding next action: {e}")
            raise

    async def _extract_tool_call(self, response_text: str, user_input: str) -> Optional[Dict[str, Any]]:
        """
        Extract tool call information from LLM response using LLM-based tool selection.
        Returns a dict with keys: type, tool, args, confirmation.
        """
        try:
            # Use LLM to determine if a tool should be called
            tool_call = await self._llm_tool_selection(response_text, user_input)
            if tool_call:
                # Don't include the tool call format in the confirmation message
                # The confirmation will be handled by the ConfirmationStateHandler
                tool_call["confirmation"] = None
                return tool_call
            return None
        except Exception as e:
            logger.error(f"Error extracting tool call: {e}")
            return None

    async def _validate_and_format_tool_call(self, tool_name: str, tool_args: str, user_input: str) -> Dict[str, Any]:
        """
        Generic validation and formatting of a tool call based on the tool's signature.
        Uses the convert_natural_language_to_structured_args utility when needed.
        Returns a dict with keys: type, tool, args.
        """
        try:
            import inspect
            import json
            tool = None
            for t in self.tools:
                if t.name == tool_name:
                    tool = t
                    break
            if not tool:
                logger.error(f"Tool '{tool_name}' not found in available tools")
                return None
            
            sig = inspect.signature(tool.func)
            params = [p for p in sig.parameters if p != 'self']
            
            # Try to parse tool_args as JSON/dict first
            parsed_args = None
            if isinstance(tool_args, dict):
                parsed_args = tool_args
            else:
                try:
                    parsed_args = json.loads(tool_args)
                except Exception:
                    # If it's not valid JSON, treat as natural language
                    parsed_args = tool_args.strip()
            
            # Check if we need to convert natural language to structured args
            needs_conversion = False
            if isinstance(parsed_args, str) and len(params) > 1:
                # Multiple parameters expected but we have a string - likely needs conversion
                needs_conversion = True
            elif isinstance(parsed_args, str) and len(params) == 1:
                # Single parameter - check if it expects a dict
                param_name = params[0]
                param = sig.parameters[param_name]
                if param.annotation in [dict, Dict, Any] or param.annotation == inspect.Parameter.empty:
                    # Parameter expects a dict but we have a string - needs conversion
                    needs_conversion = True
            
            if needs_conversion:
                logger.debug(f"Converting natural language to structured args for tool '{tool_name}'")
                try:
                    # Build expected parameters dict for the utility function
                    expected_parameters = {}
                    for param_name in params:
                        param = sig.parameters[param_name]
                        param_info = {
                            'type': param.annotation if param.annotation != inspect.Parameter.empty else Any,
                            'required': param.default == inspect.Parameter.empty,
                            'default': param.default if param.default != inspect.Parameter.empty else None
                        }
                        expected_parameters[param_name] = param_info
                    
                    # Use the utility function to convert natural language to structured args
                    converted_args = await convert_natural_language_to_structured_args(
                        self.llm, tool_name, parsed_args, expected_parameters
                    )
                    
                    # Build the final args dict
                    if len(params) == 1:
                        param_name = params[0]
                        args = {param_name: converted_args}
                    else:
                        args = converted_args
                    
                    return {
                        "type": "tool_call",
                        "tool": tool_name,
                        "args": args
                    }
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

    async def _llm_tool_selection(self, response_text: str, user_input: str) -> Optional[Dict[str, Any]]:
        """
        Use LLM to determine if a tool should be called based on the response and user input.
        
        Args:
            response_text: The LLM's response text
            user_input: The original user input
            
        Returns:
            Dict containing tool call details or None if no tool call should be made
        """
        try:
            # First, check if the response already contains a tool call in the expected format
            # Look for patterns like: tool_name: "arguments" or TOOL_CALL: tool_name arguments
            for tool in self.tools:
                # Check for tool_name: "arguments" format
                prefix = f"{tool.name}:"
                if prefix in response_text:
                    # Extract the tool call
                    parts = response_text.split(prefix, 1)
                    if len(parts) >= 2:
                        tool_args = parts[1].strip()
                        # Remove quotes if present
                        if tool_args.startswith('"') and tool_args.endswith('"'):
                            tool_args = tool_args[1:-1]
                        return await self._validate_and_format_tool_call(tool.name, tool_args, user_input)
            
            # Check for TOOL_CALL: format as fallback
            if "TOOL_CALL:" in response_text:
                tool_call_line = response_text.split("TOOL_CALL:")[1].strip().split("\n")[0]
                parts = tool_call_line.split(" ", 1)
                if len(parts) >= 2:
                    tool_name = parts[0].strip().rstrip(":")
                    tool_args = parts[1].strip()
                    return await self._validate_and_format_tool_call(tool_name, tool_args, user_input)
            
            # If no explicit tool call found, use LLM to determine if one should be made
            tool_descriptions = "\n".join([
                f"- {tool.name}: {tool.description}" for tool in self.tools
            ])
            
            prompt = f"""
            Based on the user's input and the AI's response, determine if a tool should be called.
            
            User input: {user_input}
            AI response: {response_text}
            
            Available tools:
            {tool_descriptions}
            
            If a tool should be called, respond with:
            TOOL_CALL: <tool_name> <arguments>
            
            If no tool should be called, respond with: NO_TOOL
            
            Examples:
            1. User: "Schedule a workout for tomorrow"
               AI: "I'll schedule a workout for you tomorrow."
               Response: TOOL_CALL: create_calendar_event "schedule a workout for tomorrow"
            
            2. User: "What's on my calendar today?"
               AI: "Let me check your calendar for today."
               Response: TOOL_CALL: get_calendar_events "today"
            
            3. User: "Hello"
               AI: "Hello! How can I help you today?"
               Response: NO_TOOL
            """
            
            messages = [
                SystemMessage(content="You are a tool selection assistant. Determine if a tool should be called based on the user input and AI response."),
                HumanMessage(content=prompt)
            ]
            
            try:
                response = await asyncio.wait_for(
                    self.llm.ainvoke(messages),
                    timeout=10.0  # 10 second timeout
                )
            except asyncio.TimeoutError:
                logger.error("LLM tool selection timed out")
                return None
            
            # Handle both string and AIMessage responses
            if isinstance(response, str):
                response_text = response
            elif hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
            
            response_text = response_text.strip()
            
            # Check if a tool should be called
            if "TOOL_CALL:" in response_text:
                tool_call_line = response_text.split("TOOL_CALL:")[1].strip().split("\n")[0]
                parts = tool_call_line.split(" ", 1)
                if len(parts) >= 2:
                    tool_name = parts[0].strip().rstrip(":")
                    tool_args = parts[1].strip()
                    return await self._validate_and_format_tool_call(tool_name, tool_args, user_input)
            
            return None
            
        except Exception as e:
            logger.error(f"Error in LLM tool selection: {e}")
            return None

    def _convert_message(self, msg):
        """Convert various message formats to LangChain format."""
        from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
        if isinstance(msg, BaseMessage):
            return msg
        if isinstance(msg, dict):
            if msg.get('role') == 'user':
                return HumanMessage(content=msg.get('content', ''))
            elif msg.get('role') == 'assistant':
                return AIMessage(content=msg.get('content', ''))
            else:
                return HumanMessage(content=str(msg))
        elif isinstance(msg, str):
            return HumanMessage(content=msg)
        else:
            return HumanMessage(content=str(msg))

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
            user_input = context.get('user_input', '').lower()
            if 'yes' in user_input or 'confirm' in user_input or 'sure' in user_input:
                return 'confirmed'
            elif 'no' in user_input or 'cancel' in user_input:
                return 'cancelled'
            else:
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
                                    agent_state=None) -> AsyncGenerator[str, None]:
        """
        Process messages using the advanced state machine with transition graph validation.
        
        Args:
            messages: List of messages to process
            execute_tool_func: Function to execute tools
            get_tool_confirmation_func: Function to get tool confirmation messages
            summarize_tool_result_func: Function to summarize tool results
            agent_state: Current state of the agent
            
        Yields:
            str: Response messages
        """
        try:
            # Convert messages to LangChain format
            input_messages = []
            for msg in messages:
                converted = self._convert_message(msg)
                if converted:
                    input_messages.append(converted)
            
            if not input_messages:
                yield "I didn't receive any valid messages to process."
                return
            
            # Get the last user message
            user_message = input_messages[-1]
            if not isinstance(user_message, HumanMessage):
                yield "I need a user message to process."
                return
            
            # Initialize context
            context = {
                'history': [user_message.content],
                'user_input': user_message.content,
                'agent_state': agent_state,
                'execute_tool_func': execute_tool_func,
                'get_tool_confirmation_func': get_tool_confirmation_func,
                'summarize_tool_result_func': summarize_tool_result_func,
                'agent_action': None,
                'tool_result': None,
                'last_tool': None,
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