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
from backend.time_formatting import extract_timeframe_from_text

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
                                    summarize_tool_result_func) -> AsyncGenerator[str, None]:
        """
        Process messages and return a streaming response with multi-step tool execution.
        
        Args:
            messages: List of messages to process
            execute_tool_func: Function to execute tools
            get_tool_confirmation_func: Function to get tool confirmation messages
            summarize_tool_result_func: Function to summarize tool results
            
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
                    agent_action = await self.decide_next_action(history)
                    if agent_action["type"] == "message":
                        yield agent_action["content"]
                        state = "DONE"
                    elif agent_action["type"] == "tool_call":
                        last_tool = agent_action["tool"]
                        # Send confirmation message before calling tool
                        confirmation_message = await get_tool_confirmation_func(last_tool, agent_action["args"])
                        yield confirmation_message
                        state = "AGENT_TOOL_CALL"
                    else:
                        state = "DONE"
                elif state == "AGENT_TOOL_CALL":
                    tool_result = await execute_tool_func(agent_action["tool"], agent_action["args"])
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
            
            # Preference detection (integrated)
            preference = await self.extract_preference_func(input_text)
            if preference:
                return {
                    "type": "tool_call",
                    "tool": "add_preference_to_kg",
                    "args": preference
                }

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
            response = await self.llm.ainvoke(messages)
            
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
            
            # Check if the response contains a tool call
            if "TOOL_CALL:" in response_text:
                tool_call_line = response_text.split("TOOL_CALL:")[1].strip().split("\n")[0]
                parts = tool_call_line.split(" ", 1)
                if len(parts) >= 2:
                    tool_name = parts[0].strip().rstrip(":")
                    tool_args = parts[1].strip()
                    # If get_calendar_events, override args with timeframe if present in user message
                    if tool_name == "get_calendar_events":
                        timeframe = self.extract_timeframe_func(input_text)
                        if timeframe:
                            tool_args = f'"{timeframe}"'
                    logger.info(f"[TOOL_CALL] Tool selected: {tool_name}, Args: {tool_args}")
                    return {
                        "type": "tool_call",
                        "tool": tool_name,
                        "args": tool_args
                    }
            # Fallback: detect lines like 'find_nearby_workout_locations: ...' as tool calls
            for tool in self.tools:
                prefix = f"{tool.name}:"
                if response_text.strip().startswith(prefix):
                    tool_args = response_text.strip()[len(prefix):].strip()
                    return {
                        "type": "tool_call",
                        "tool": tool.name,
                        "args": tool_args
                    }
            
            # Handle regular messages
            return {
                "type": "message",
                "content": response_text
            }
                
        except Exception as e:
            logger.error(f"Error deciding next action: {e}")
            raise

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