"""
Tool configuration and metadata for the Personal Trainer Tool Manager.

This module contains declarative tool definitions and metadata that can be
easily modified without changing the core tool management logic.
"""

from typing import Dict, Any, List
from backend.tools.preferences_tools import add_preference_to_kg

# Tool metadata for declarative registration
TOOL_METADATA = {
    'calendar': {
        'get_upcoming_events': {
            'name': 'get_calendar_events',
            'description': 'Get upcoming calendar events',
            'category': 'calendar',
            'examples': [
                'Get events for this week',
                'Show my calendar for tomorrow',
                'What meetings do I have today?'
            ]
        },
        'write_event': {
            'name': 'create_calendar_event',
            'description': 'Create a new calendar event',
            'category': 'calendar',
            'examples': [
                'Schedule a workout for tomorrow at 6pm',
                'Create a meeting with John on Friday at 2pm',
                'Add a gym session to my calendar'
            ]
        },
        'delete_events_in_range': {
            'name': 'delete_events_in_range',
            'description': 'Delete all calendar events within a specified time range',
            'category': 'calendar',
            'examples': [
                'Clear my calendar for this afternoon',
                'Delete all events for tomorrow',
                'Remove meetings from 2pm to 4pm today'
            ]
        }
    },
    'gmail': {
        'send_message': {
            'name': 'send_email',
            'description': 'Send an email to a recipient',
            'category': 'communication',
            'examples': [
                'Send an email to john@example.com about our workout session',
                'Email my trainer about my progress',
                'Send a reminder to myself about tomorrow\'s workout'
            ]
        },
        'get_recent_emails': {
            'name': 'get_recent_emails',
            'description': 'Get recent emails from Gmail',
            'category': 'communication',
            'examples': [
                'Show me recent emails from my trainer',
                'Get emails from this week',
                'Check for any fitness-related emails'
            ]
        }
    },
    'tasks': {
        'create_task': {
            'name': 'create_task',
            'description': 'Create a new task',
            'category': 'productivity',
            'examples': [
                'Create a task to buy protein powder',
                'Add a reminder to schedule my next workout',
                'Create a task to track my weight this week'
            ]
        },
        'get_tasks': {
            'name': 'get_tasks',
            'description': 'Get tasks from the task list',
            'category': 'productivity',
            'examples': [
                'Show me my pending tasks',
                'List all my fitness-related tasks',
                'What tasks do I have for today?'
            ]
        }
    },
    'drive': {
        'search_files': {
            'name': 'search_drive',
            'description': 'Search for files in Google Drive',
            'category': 'storage',
            'examples': [
                'Find my workout plans in Drive',
                'Search for nutrition documents',
                'Look for my fitness tracking spreadsheet'
            ]
        },
        'create_folder': {
            'name': 'create_folder',
            'description': 'Create a new folder in Google Drive',
            'category': 'storage',
            'examples': [
                'Create a folder for my workout photos',
                'Make a new folder for nutrition plans',
                'Create a fitness documents folder'
            ]
        }
    },
    'sheets': {
        'get_sheet_data': {
            'name': 'get_sheet_data',
            'description': 'Get data from a Google Sheet',
            'category': 'data',
            'examples': [
                'Get my workout data from the tracker',
                'Show me my nutrition entries',
                'Retrieve my fitness progress data'
            ]
        },
        'create_workout_tracker': {
            'name': 'create_workout_tracker',
            'description': 'Create a new workout tracking spreadsheet',
            'category': 'fitness',
            'examples': [
                'Create a new workout tracker for this month',
                'Set up a fitness tracking spreadsheet',
                'Make a new workout log'
            ]
        },
        'add_workout_entry': {
            'name': 'add_workout_entry',
            'description': 'Add a workout entry to the tracker',
            'category': 'fitness',
            'examples': [
                'Log today\'s workout: 30 minutes cardio',
                'Add my strength training session',
                'Record my yoga class'
            ]
        },
        'add_nutrition_entry': {
            'name': 'add_nutrition_entry',
            'description': 'Add a nutrition entry to the tracker',
            'category': 'fitness',
            'examples': [
                'Log my breakfast: oatmeal and banana',
                'Add my lunch: chicken salad',
                'Record my dinner calories'
            ]
        }
    },
    'maps': {
        'get_directions': {
            'name': 'get_directions',
            'description': 'Get directions between two locations',
            'category': 'location',
            'examples': [
                'Get directions to the gym',
                'How do I get to the park for my run?',
                'Directions to the yoga studio'
            ]
        },
        'find_nearby_workout_locations': {
            'name': 'get_nearby_locations',
            'description': 'Find nearby workout locations like gyms, parks, etc.',
            'category': 'fitness',
            'examples': [
                'Find gyms near my location',
                'Show me nearby parks for running',
                'Find yoga studios in my area'
            ]
        },
        'find_nearby_places': {
            'name': 'get_nearby_places',
            'description': 'Find nearby places of interest',
            'category': 'location',
            'examples': [
                'Find restaurants near the gym',
                'Show me nearby coffee shops',
                'Find parking near the park'
            ]
        }
    }
}

# Custom tools that don't belong to a specific service
CUSTOM_TOOLS = {
    'add_preference_to_kg': {
        'func': add_preference_to_kg,
        'description': 'Add a user preference to the knowledge graph',
        'category': 'knowledge',
        'examples': [
            'Remember that I prefer morning workouts',
            'Save that I like strength training',
            'Note that I\'m allergic to nuts'
        ]
    }
}

# Tool categories with descriptions
TOOL_CATEGORIES = {
    'calendar': {
        'name': 'Calendar Management',
        'description': 'Tools for managing calendar events and scheduling',
        'icon': '📅'
    },
    'communication': {
        'name': 'Communication',
        'description': 'Tools for email and messaging',
        'icon': '📧'
    },
    'productivity': {
        'name': 'Productivity',
        'description': 'Tools for task management and organization',
        'icon': '✅'
    },
    'storage': {
        'name': 'File Storage',
        'description': 'Tools for managing files and folders',
        'icon': '📁'
    },
    'data': {
        'name': 'Data Management',
        'description': 'Tools for working with spreadsheets and data',
        'icon': '📊'
    },
    'fitness': {
        'name': 'Fitness Tracking',
        'description': 'Tools specifically for fitness and health tracking',
        'icon': '💪'
    },
    'location': {
        'name': 'Location Services',
        'description': 'Tools for finding places and getting directions',
        'icon': '📍'
    },
    'knowledge': {
        'name': 'Knowledge Management',
        'description': 'Tools for managing user preferences and knowledge',
        'icon': '🧠'
    }
}

def get_tool_by_name(tool_name: str) -> Dict[str, Any]:
    """Get tool metadata by name."""
    # Search in service tools
    for service_name, service_metadata in TOOL_METADATA.items():
        for method_name, tool_info in service_metadata.items():
            if tool_info['name'] == tool_name:
                return {**tool_info, 'service': service_name, 'method': method_name}
    
    # Search in custom tools
    if tool_name in CUSTOM_TOOLS:
        return {**CUSTOM_TOOLS[tool_name], 'service': 'custom'}
    
    return None

def get_tools_by_category(category: str) -> List[Dict[str, Any]]:
    """Get all tools in a specific category."""
    tools = []
    
    # Search in service tools
    for service_name, service_metadata in TOOL_METADATA.items():
        for method_name, tool_info in service_metadata.items():
            if tool_info.get('category') == category:
                tools.append({**tool_info, 'service': service_name, 'method': method_name})
    
    # Search in custom tools
    for tool_name, tool_info in CUSTOM_TOOLS.items():
        if tool_info.get('category') == category:
            tools.append({**tool_info, 'service': 'custom'})
    
    return tools

def get_all_tool_names() -> List[str]:
    """Get all available tool names."""
    tool_names = []
    
    # Get service tool names
    for service_metadata in TOOL_METADATA.values():
        for tool_info in service_metadata.values():
            tool_names.append(tool_info['name'])
    
    # Get custom tool names
    tool_names.extend(CUSTOM_TOOLS.keys())
    
    return tool_names

def validate_tool_configuration() -> List[str]:
    """Validate the tool configuration and return any issues."""
    issues = []
    
    # Check for duplicate tool names
    tool_names = get_all_tool_names()
    duplicates = [name for name in set(tool_names) if tool_names.count(name) > 1]
    if duplicates:
        issues.append(f"Duplicate tool names found: {duplicates}")
    
    # Check for invalid categories
    valid_categories = set(TOOL_CATEGORIES.keys())
    for service_metadata in TOOL_METADATA.values():
        for tool_info in service_metadata.values():
            category = tool_info.get('category')
            if category and category not in valid_categories:
                issues.append(f"Invalid category '{category}' for tool '{tool_info['name']}'")
    
    for tool_info in CUSTOM_TOOLS.values():
        category = tool_info.get('category')
        if category and category not in valid_categories:
            issues.append(f"Invalid category '{category}' for custom tool")
    
    return issues 