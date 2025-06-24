"""
Agent orchestration package for the Personal Trainer Agent.

This package contains the core agent orchestration logic including:
- Agent state management
- State machine for agent workflow
- State handlers for different agent states
- Orchestrated agent implementation
- Auto-discovery tool management
"""

from .agent_state import AgentState
from .agent_state_machine import AgentStateMachine, AgentTransitionMachine
from .state_handler import (
    AgentState as StateHandlerAgentState,
    StateHandler,
    ThinkingStateHandler,
    ToolCallStateHandler,
    SummarizeToolResultStateHandler,
    StateTransitionGraph
)
from .orchestrated_agent import OrchestratedAgent
from .auto_tool_manager import (
    AutoToolManager,
    ToolMetadata,
    ToolDiscoveryStrategy,
    MetadataBasedDiscovery,
    ReflectionBasedDiscovery
)

__all__ = [
    'AgentState',
    'AgentStateMachine',
    'AgentTransitionMachine',
    'StateHandlerAgentState',
    'StateHandler',
    'ThinkingStateHandler',
    'ToolCallStateHandler',
    'SummarizeToolResultStateHandler',
    'StateTransitionGraph',
    'OrchestratedAgent',
    'AutoToolManager',
    'ToolMetadata',
    'ToolDiscoveryStrategy',
    'MetadataBasedDiscovery',
    'ReflectionBasedDiscovery',
] 