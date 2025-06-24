"""
Tools package for the Personal Trainer Agent.

This package contains all tool-related functionality including:
- Personal trainer specific tool manager
- Tool configuration and metadata
- Individual tool implementations
"""

from backend.tools.personal_trainer_tool_manager import PersonalTrainerToolManager
from backend.tools.tool_config import (
    TOOL_METADATA,
    CUSTOM_TOOLS,
    TOOL_CATEGORIES,
    get_tool_by_name,
    get_tools_by_category,
    get_all_tool_names,
    validate_tool_configuration
)
from backend.tools.preferences_tools import add_preference_to_kg

__all__ = [
    # Tool managers
    'PersonalTrainerToolManager',
    
    # Configuration
    'TOOL_METADATA',
    'CUSTOM_TOOLS',
    'TOOL_CATEGORIES',
    'get_tool_by_name',
    'get_tools_by_category',
    'get_all_tool_names',
    'validate_tool_configuration',
    
    # Individual tools
    'add_preference_to_kg',
] 