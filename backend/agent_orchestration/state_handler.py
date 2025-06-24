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
                    await context['agent_state'].update(status=AgentState.TOOL_CALL.value, last_tool_result=None)
                confirmation_message = agent_action.get("confirmation") or "I'll proceed with your request."
                return AgentState.TOOL_CALL, confirmation_message
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


class StateTransitionGraph:
    """Represents a state transition graph for the agent state machine."""
    
    def __init__(self):
        self.transitions = {
            AgentState.THINKING: {
                'message_response': AgentState.DONE,
                'tool_call': AgentState.TOOL_CALL,
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