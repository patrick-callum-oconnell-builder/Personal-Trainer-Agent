"""
State Handlers for Agent State Machine

This module contains the state handler classes that implement the logic for each
state in the agent state machine. Each handler is responsible for processing
a specific state and determining the next state transition.

The handlers work with AgentState as the single source of truth for all
conversation data, focusing on state transitions rather than data management.
"""

import logging
from typing import Dict, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Enumeration of possible agent states."""
    THINKING = "AGENT_THINKING"
    CONFIRMATION = "AGENT_CONFIRMATION"
    TOOL_CALL = "AGENT_TOOL_CALL"
    SUMMARIZE_TOOL_RESULT = "AGENT_SUMMARIZE_TOOL_RESULT"
    DONE = "DONE"
    ERROR = "ERROR"


class StateHandler:
    """Base class for state handlers."""
    
    async def handle(self, context: Dict[str, Any]) -> tuple[AgentState, Optional[str]]:
        """
        Handle the current state and return the next state and optional response.
        
        Args:
            context: Current state context containing:
                - agent_state: AgentState object (single source of truth)
                - execute_tool_func: Function to execute tools
                - get_tool_confirmation_func: Function to get tool confirmation messages
                - summarize_tool_result_func: Function to summarize tool results
            
        Returns:
            Tuple of (next_state, optional_response)
        """
        raise NotImplementedError


class ThinkingStateHandler(StateHandler):
    """Handler for the AGENT_THINKING state."""
    
    def __init__(self, state_machine):
        self.state_machine = state_machine
    
    async def handle(self, context: Dict[str, Any]) -> tuple[AgentState, Optional[str]]:
        """Handle the thinking state - decide next action."""
        agent_state = context['agent_state']
        
        # Update agent state status
        await agent_state.update(status=AgentState.THINKING.value)
        
        try:
            # Use the state machine to decide next action based on agent state
            agent_action = await self.state_machine.decide_next_action(agent_state)
            
            if agent_action["type"] == "message":
                # Add the AI response to the conversation history
                from langchain_core.messages import AIMessage
                new_messages = agent_state.messages + [AIMessage(content=agent_action["content"])]
                await agent_state.update(messages=new_messages, status="awaiting_user")
                return AgentState.DONE, agent_action["content"]
                
            elif agent_action["type"] == "tool_call":
                # Store the tool action in agent state for later use
                await agent_state.update(
                    status=AgentState.CONFIRMATION.value, 
                    last_tool_result=None
                )
                # Store tool action in context for state handlers
                context['agent_action'] = agent_action
                context['last_tool'] = agent_action["tool"]
                return AgentState.CONFIRMATION, None
            else:
                await agent_state.update(status="error")
                return AgentState.ERROR, "Invalid action type returned from decision making."
                
        except Exception as e:
            logger.error(f"Error in ThinkingStateHandler: {e}")
            await agent_state.update(status=AgentState.ERROR.value)
            return AgentState.ERROR, f"Sorry, something went wrong while deciding next action: {str(e)}"


class ConfirmationStateHandler(StateHandler):
    """Handler for the AGENT_CONFIRMATION state."""
    
    async def handle(self, context: Dict[str, Any]) -> tuple[AgentState, Optional[str]]:
        """Handle the confirmation state - send confirmation message and proceed to tool execution."""
        agent_state = context['agent_state']
        agent_action = context.get('agent_action')
        
        if not agent_action:
            await agent_state.update(status="error")
            return AgentState.ERROR, "No tool action found for confirmation."
        
        # Generate a user-friendly confirmation message
        tool_name = agent_action["tool"]
        tool_args = agent_action["args"]
        
        # Create a clean confirmation message without exposing internal tool names
        if tool_name == "create_calendar_event":
            confirmation = "I'll schedule that for you."
        elif tool_name == "get_calendar_events":
            confirmation = "Let me check your calendar."
        elif tool_name == "send_email":
            confirmation = "I'll send that email for you."
        elif tool_name == "create_task":
            confirmation = "I'll create that task for you."
        elif tool_name == "search_location":
            confirmation = "I'll search for that location."
        elif tool_name == "create_workout_tracker":
            confirmation = "I'll create a workout tracker for you."
        elif tool_name == "add_workout_entry":
            confirmation = "I'll log that workout for you."
        elif tool_name == "add_nutrition_entry":
            confirmation = "I'll log that nutrition entry for you."
        else:
            confirmation = "I'll handle that for you."
        
        # Add the confirmation message to conversation history
        from langchain_core.messages import AIMessage
        new_messages = agent_state.messages + [AIMessage(content=confirmation)]
        await agent_state.update(messages=new_messages)
        
        return AgentState.TOOL_CALL, confirmation


class ToolCallStateHandler(StateHandler):
    """Handler for the AGENT_TOOL_CALL state."""
    
    async def handle(self, context: Dict[str, Any]) -> tuple[AgentState, Optional[str]]:
        """Handle the tool call state - execute tool."""
        agent_state = context['agent_state']
        agent_action = context.get('agent_action')
        
        if not agent_action:
            await agent_state.update(status="error")
            return AgentState.ERROR, "No tool action found for execution."
        
        try:
            # Execute the tool
            tool_result = await context['execute_tool_func'](
                agent_action["tool"], 
                agent_action["args"]
            )
            
            # Store the tool result in agent state
            await agent_state.update(
                status=AgentState.SUMMARIZE_TOOL_RESULT.value,
                last_tool_result=tool_result
            )
            
            # Store tool result in context for state handlers
            context['tool_result'] = tool_result
            
            return AgentState.SUMMARIZE_TOOL_RESULT, None
            
        except Exception as e:
            logger.error(f"Error in ToolCallStateHandler: {e}")
            await agent_state.update(status=AgentState.ERROR.value)
            return AgentState.ERROR, f"Sorry, something went wrong while executing the tool: {str(e)}"


class SummarizeToolResultStateHandler(StateHandler):
    """Handler for the AGENT_SUMMARIZE_TOOL_RESULT state."""
    
    async def handle(self, context: Dict[str, Any]) -> tuple[AgentState, Optional[str]]:
        """Handle the summarize state - summarize tool result."""
        agent_state = context['agent_state']
        last_tool = context.get('last_tool')
        tool_result = context.get('tool_result')
        
        if not tool_result:
            await agent_state.update(status="error")
            return AgentState.ERROR, "No tool result found for summarization."
        
        try:
            # Generate summary using the provided function
            summary = await context['summarize_tool_result_func'](
                last_tool, 
                tool_result
            )
            
            if not summary:
                logger.error(f"LLM returned empty summary for tool {last_tool} and result {tool_result}")
                raise RuntimeError("LLM returned empty summary")
            
            # Add the summary to conversation history
            from langchain_core.messages import AIMessage
            new_messages = agent_state.messages + [AIMessage(content=summary)]
            await agent_state.update(
                messages=new_messages,
                status="awaiting_user"
            )
            
            return AgentState.DONE, summary
            
        except Exception as e:
            logger.error(f"Error in SummarizeToolResultStateHandler: {e}")
            await agent_state.update(status=AgentState.ERROR.value)
            return AgentState.ERROR, f"Sorry, something went wrong while summarizing the result: {str(e)}"


class ErrorStateHandler(StateHandler):
    """Handler for the ERROR state."""
    
    async def handle(self, context: Dict[str, Any]) -> tuple[AgentState, Optional[str]]:
        """Handle the error state."""
        agent_state = context['agent_state']
        
        # Ensure agent state is marked as error
        await agent_state.update(status="error")
        
        return AgentState.DONE, "An error occurred. Please try again."


class StateTransitionGraph:
    """
    Graph representing valid state transitions in the agent state machine.
    
    This class defines the allowed transitions between states and provides
    validation for state transitions.
    """
    
    def __init__(self):
        """Initialize the transition graph with valid state transitions."""
        self.transitions = {
            AgentState.THINKING: {
                'message_response': AgentState.DONE,
                'tool_call': AgentState.CONFIRMATION,
                'error': AgentState.ERROR,
            },
            AgentState.CONFIRMATION: {
                'confirmed': AgentState.TOOL_CALL,
                'cancelled': AgentState.DONE,
                'error': AgentState.ERROR,
            },
            AgentState.TOOL_CALL: {
                'success': AgentState.SUMMARIZE_TOOL_RESULT,
                'error': AgentState.ERROR,
            },
            AgentState.SUMMARIZE_TOOL_RESULT: {
                'success': AgentState.DONE,
                'error': AgentState.ERROR,
            },
            AgentState.ERROR: {
                'error': AgentState.DONE,
            },
        }
    
    def get_next_state(self, current_state: AgentState, event: str) -> Optional[AgentState]:
        """
        Get the next state based on the current state and event.
        
        Args:
            current_state: Current state
            event: Event that occurred
            
        Returns:
            Next state or None if transition is invalid
        """
        state_transitions = self.transitions.get(current_state, {})
        return state_transitions.get(event) 