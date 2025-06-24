"""
State Handlers for Agent State Machine

This module contains the state handler classes that implement the logic for each
state in the agent state machine. Each handler is responsible for processing
a specific state and determining the next state transition.
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
            context: Current state context
            
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
        if context.get('agent_state'):
            await context['agent_state'].update(status=AgentState.THINKING.value)
        try:
            agent_action = await self.state_machine.decide_next_action(context['history'])
            if agent_action["type"] == "message":
                return AgentState.DONE, agent_action["content"]
            elif agent_action["type"] == "tool_call":
                context['last_tool'] = agent_action["tool"]
                context['agent_action'] = agent_action
                if context.get('agent_state'):
                    await context['agent_state'].update(status=AgentState.CONFIRMATION.value, last_tool_result=None)
                # Transition to confirmation state instead of directly to tool call
                return AgentState.CONFIRMATION, None
            else:
                return AgentState.DONE, None
        except Exception as e:
            logger.error(f"Error in ThinkingStateHandler: {e}")
            if context.get('agent_state'):
                await context['agent_state'].update(status=AgentState.ERROR.value)
            return AgentState.ERROR, f"Sorry, something went wrong while deciding next action: {str(e)}"


class ToolCallStateHandler(StateHandler):
    """Handler for the AGENT_TOOL_CALL state."""
    
    async def handle(self, context: Dict[str, Any]) -> tuple[AgentState, Optional[str]]:
        """Handle the tool call state - execute tool."""
        try:
            tool_result = await context['execute_tool_func'](
                context['agent_action']["tool"], 
                context['agent_action']["args"]
            )
            if context.get('agent_state'):
                await context['agent_state'].update(last_tool_result=tool_result)
            context['history'].append(f"TOOL RESULT: {tool_result}")
            context['tool_result'] = tool_result
            return AgentState.SUMMARIZE_TOOL_RESULT, None
        except Exception as e:
            logger.error(f"Error in ToolCallStateHandler: {e}")
            if context.get('agent_state'):
                await context['agent_state'].update(status=AgentState.ERROR.value)
            return AgentState.ERROR, f"Sorry, something went wrong while executing the tool: {str(e)}"


class SummarizeToolResultStateHandler(StateHandler):
    """Handler for the AGENT_SUMMARIZE_TOOL_RESULT state."""
    
    async def handle(self, context: Dict[str, Any]) -> tuple[AgentState, Optional[str]]:
        """Handle the summarize state - summarize tool result."""
        try:
            summary = await context['summarize_tool_result_func'](
                context['last_tool'], 
                context['tool_result']
            )
            if not summary:
                logger.error(f"LLM returned empty summary for tool {context['last_tool']} and result {context['tool_result']}")
                raise RuntimeError("LLM returned empty summary")
            if context.get('agent_state'):
                await context['agent_state'].update(status=AgentState.DONE.value)
            return AgentState.DONE, summary
        except Exception as e:
            logger.error(f"Error in SummarizeToolResultStateHandler: {e}")
            if context.get('agent_state'):
                await context['agent_state'].update(status=AgentState.ERROR.value)
            return AgentState.ERROR, f"Sorry, something went wrong while summarizing the tool result: {str(e)}"


class ErrorStateHandler(StateHandler):
    """Handler for the ERROR state."""
    async def handle(self, context: Dict[str, Any]) -> tuple[AgentState, Optional[str]]:
        return AgentState.DONE, "Sorry, an error occurred and the agent cannot continue this conversation."


class ConfirmationStateHandler(StateHandler):
    """Handler for the AGENT_CONFIRMATION state."""
    async def handle(self, context: Dict[str, Any]) -> tuple[AgentState, Optional[str]]:
        """Send an informational confirmation message and proceed to tool execution."""
        try:
            # Generate an informational confirmation message for the user
            agent_action = context.get('agent_action')
            if not agent_action:
                return AgentState.ERROR, "Sorry, I couldn't determine what to confirm."
            tool = agent_action["tool"]
            args = agent_action["args"]
            
            # Create a clean, user-friendly confirmation message without tool names
            if tool == "create_calendar_event":
                confirmation_message = "Sure, I'll add that to your calendar."
            elif tool == "get_calendar_events":
                confirmation_message = "I'll check your calendar for that time period."
            elif tool == "send_email":
                confirmation_message = "I'll send that email for you."
            elif tool == "create_task":
                confirmation_message = "I'll create that task for you."
            elif tool == "get_tasks":
                confirmation_message = "I'll get your tasks for you."
            elif tool == "search_drive":
                confirmation_message = "I'll search your Drive for that."
            elif tool == "create_folder":
                confirmation_message = "I'll create that folder for you."
            elif tool == "create_workout_tracker":
                confirmation_message = "I'll create a workout tracker for you."
            elif tool == "add_workout_entry":
                confirmation_message = "I'll log that workout for you."
            elif tool == "add_nutrition_entry":
                confirmation_message = "I'll log that nutrition entry for you."
            elif tool == "get_directions":
                confirmation_message = "I'll get those directions for you."
            elif tool == "get_nearby_locations":
                confirmation_message = "I'll find nearby locations for you."
            elif tool == "add_preference_to_kg":
                confirmation_message = "I'll remember that preference for you."
            elif tool == "resolve_calendar_conflict":
                confirmation_message = "I'll help resolve that calendar conflict."
            else:
                confirmation_message = "I'll take care of that for you."
            
            if context.get('agent_state'):
                await context['agent_state'].update(status=AgentState.CONFIRMATION.value)
            
            # Immediately proceed to tool execution after confirming
            return AgentState.TOOL_CALL, confirmation_message
        except Exception as e:
            logger.error(f"Error in ConfirmationStateHandler: {e}")
            if context.get('agent_state'):
                await context['agent_state'].update(status=AgentState.ERROR.value)
            return AgentState.ERROR, f"Sorry, something went wrong while confirming the action: {str(e)}"


class StateTransitionGraph:
    """Represents a state transition graph for the agent state machine."""
    
    def __init__(self):
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
                'recover': AgentState.THINKING,
                'end': AgentState.DONE,
            },
            AgentState.DONE: {
                'end': None,  # Terminal state
            }
        }
    
    def get_next_state(self, current_state: AgentState, event: str) -> Optional[AgentState]:
        """
        Get the next state based on current state and event.
        
        Args:
            current_state: Current state
            event: Event that occurred
            
        Returns:
            Next state or None if terminal
        """
        state_transitions = self.transitions.get(current_state, {})
        return state_transitions.get(event) 