"""
Agent State Machine for Personal Trainer AI

This module contains the state machine logic for orchestrating agent conversations,
including decision-making, tool execution, and response generation.
"""

import json
import logging
from typing import List, Dict, Any, Optional, Union, AsyncGenerator
from datetime import datetime, timezone as dt_timezone
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.tools import Tool
from backend.prompts import get_system_prompt
from backend.utilities.time_formatting import extract_timeframe_from_text
import asyncio

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
    
    async def process_messages_stream(self, messages: List[BaseMessage], 
                                    execute_tool_func, 
                                    get_tool_confirmation_func,
                                    summarize_tool_result_func,
                                    agent_state=None) -> AsyncGenerator[str, None]:
        """
        Process messages and return a streaming response with multi-step tool execution.
        
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
            
            # Multi-step process: decide action → confirm → execute → summarize
            state = "AGENT_THINKING"
            history = [user_message.content]
            agent_action = None
            tool_result = None
            last_tool = None

            while state != "DONE":
                if state == "AGENT_THINKING":
                    # Update state to thinking
                    if agent_state:
                        await agent_state.update(status="active")
                    
                    agent_action = await self.decide_next_action(history)
                    if agent_action["type"] == "message":
                        yield agent_action["content"]
                        state = "DONE"
                    elif agent_action["type"] == "tool_call":
                        last_tool = agent_action["tool"]
                        # Update state to awaiting tool
                        if agent_state:
                            await agent_state.update(status="awaiting_tool", last_tool_result=None)
                        
                        # Send confirmation message before calling tool
                        confirmation_message = await get_tool_confirmation_func(last_tool, agent_action["args"])
                        yield confirmation_message
                        state = "AGENT_TOOL_CALL"
                    else:
                        state = "DONE"
                elif state == "AGENT_TOOL_CALL":
                    tool_result = await execute_tool_func(agent_action["tool"], agent_action["args"])
                    # Update state with tool result
                    if agent_state:
                        await agent_state.update(last_tool_result=tool_result)
                    
                    # Add the tool result as a message in the history
                    history.append(f"TOOL RESULT: {tool_result}")
                    # Always go to summarize state after a tool call
                    state = "AGENT_SUMMARIZE_TOOL_RESULT"
                elif state == "AGENT_SUMMARIZE_TOOL_RESULT":
                    # Always require the LLM to summarize the tool result for the user
                    summary = await summarize_tool_result_func(last_tool, tool_result)
                    if not summary:
                        logger.error(f"LLM returned empty summary for tool {last_tool} and result {tool_result}")
                        raise RuntimeError("LLM returned empty summary")
                    
                    # Update state to done
                    if agent_state:
                        await agent_state.update(status="done")
                    
                    yield summary
                    state = "DONE"
                
        except Exception as e:
            logger.error(f"Error in process_messages_stream: {e}")
            yield f"Error processing messages: {str(e)}"

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
        Extract tool call information from LLM response using multiple strategies.
        
        Args:
            response_text: The LLM's response text
            user_input: The original user input
            
        Returns:
            Dict containing tool call details or None if no tool call found
        """
        try:
            # Strategy 1: Check for explicit TOOL_CALL format
            if "TOOL_CALL:" in response_text:
                tool_call_line = response_text.split("TOOL_CALL:")[1].strip().split("\n")[0]
                parts = tool_call_line.split(" ", 1)
                if len(parts) >= 2:
                    tool_name = parts[0].strip().rstrip(":")
                    tool_args = parts[1].strip()
                    return self._validate_and_format_tool_call(tool_name, tool_args, user_input)
            
            # Strategy 2: Check for tool name prefixes
            for tool in self.tools:
                prefix = f"{tool.name}:"
                if response_text.strip().startswith(prefix):
                    tool_args = response_text.strip()[len(prefix):].strip()
                    return self._validate_and_format_tool_call(tool.name, tool_args, user_input)
            
            # Strategy 3: Use LLM to determine if a tool should be called
            tool_call = await self._llm_tool_selection(response_text, user_input)
            if tool_call:
                return tool_call
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting tool call: {e}")
            return None

    def _validate_and_format_tool_call(self, tool_name: str, tool_args: str, user_input: str) -> Dict[str, Any]:
        """
        Validate and format a tool call with appropriate arguments.
        
        Args:
            tool_name: Name of the tool to call
            tool_args: Arguments for the tool
            user_input: Original user input for context
            
        Returns:
            Dict containing validated tool call details
        """
        # Validate that the tool exists
        if not any(tool.name == tool_name for tool in self.tools):
            logger.warning(f"Tool '{tool_name}' not found in available tools")
            return None
        
        # Special handling for specific tools
        if tool_name == "get_calendar_events":
            # Extract timeframe from user input if not provided
            if not tool_args or tool_args.strip() == '""':
                timeframe = self.extract_timeframe_func(user_input)
                if timeframe:
                    tool_args = f'"{timeframe}"'
                else:
                    tool_args = '""'
        
        logger.info(f"[TOOL_CALL] Tool selected: {tool_name}, Args: {tool_args}")
        return {
            "type": "tool_call",
            "tool": tool_name,
            "args": tool_args
        }

    async def _llm_tool_selection(self, response_text: str, user_input: str) -> Optional[Dict[str, Any]]:
        """
        Use LLM to determine if a tool should be called based on the response and user input.
        
        Args:
            response_text: The LLM's response text
            user_input: The original user input
            
        Returns:
            Dict containing tool call details or None
        """
        try:
            # Create a list of available tools with descriptions
            tool_descriptions = []
            for tool in self.tools:
                tool_descriptions.append(f"- {tool.name}: {tool.description}")
            
            prompt = f"""Based on the user's request and the assistant's response, determine if a tool should be called.

User request: {user_input}
Assistant response: {response_text}

Available tools:
{chr(10).join(tool_descriptions)}

If a tool should be called, respond with:
TOOL_CALL: tool_name "arguments"

If no tool should be called, respond with: NO_TOOL

Examples:
- User: "Schedule a workout for tomorrow at 7pm"
- Assistant: "I'll schedule that for you."
- Response: TOOL_CALL: create_calendar_event "schedule a workout for tomorrow at 7pm"

- User: "I like weightlifting"
- Assistant: "I'll remember that preference."
- Response: TOOL_CALL: add_preference_to_kg "weightlifting"

- User: "What's on my calendar tomorrow?"
- Assistant: "Let me check your calendar."
- Response: TOOL_CALL: get_calendar_events "tomorrow"

Your response:"""
            
            messages = [
                SystemMessage(content="You are an AI assistant that determines when tools should be called. Respond with either TOOL_CALL: tool_name \"arguments\" or NO_TOOL."),
                HumanMessage(content=prompt)
            ]
            
            # Add timeout to prevent hanging
            try:
                response = await asyncio.wait_for(
                    self.llm.ainvoke(messages),
                    timeout=10.0  # 10 second timeout
                )
            except asyncio.TimeoutError:
                logger.error(f"LLM call timed out in _llm_tool_selection for user input: {user_input}")
                return None
            
            content = response.content.strip() if hasattr(response, 'content') else str(response)
            
            if content.startswith("TOOL_CALL:"):
                parts = content.split(" ", 2)
                if len(parts) >= 3:
                    tool_name = parts[1].strip()
                    tool_args = parts[2].strip().strip('"')
                    return self._validate_and_format_tool_call(tool_name, tool_args, user_input)
            
            return None
            
        except Exception as e:
            logger.error(f"Error in LLM tool selection: {e}")
            return None

    def _convert_message(self, msg):
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