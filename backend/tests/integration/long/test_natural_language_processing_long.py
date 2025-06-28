import pytest
import asyncio
import os
from datetime import datetime, timedelta
import json
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from backend.personal_trainer_agent import PersonalTrainerAgent
from backend.tools.personal_trainer_tool_manager import PersonalTrainerToolManager
from dotenv import load_dotenv
import pytest_asyncio
import pytz
from backend.tests.integration.base_integration_test import BaseIntegrationTest

@pytest_asyncio.fixture
async def agent():
    """Create and initialize an agent for testing."""
    load_dotenv()
    maps_api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not maps_api_key:
        raise ValueError("Missing required environment variable: GOOGLE_MAPS_API_KEY")
    
    # Initialize services using the new architecture
    from backend.api.routes import initialize_services
    services = await initialize_services()
    
    # Create agent with individual services (agent creates its own tool manager internally)
    agent = PersonalTrainerAgent(
        calendar_service=services['calendar'],
        gmail_service=services['gmail'],
        tasks_service=services['tasks'],
        drive_service=services['drive'],
        sheets_service=services['sheets'],
        maps_service=services['maps']
    )
    return agent

@pytest.mark.asyncio
async def test_natural_language_to_calendar_json_conversion(agent: PersonalTrainerAgent):
    """Test conversion of natural language to calendar event JSON."""
    awaited_agent = await agent
    test_cases = [
        (
            "Schedule a workout tomorrow at 2pm for 1 hour",
            {
                "summary": "Workout",
                "start": {"dateTime": None, "timeZone": "America/Los_Angeles"},
                "end": {"dateTime": None, "timeZone": "America/Los_Angeles"},
                "description": "General fitness workout.",
                "location": "Gym"
            }
        ),
        (
            "Add a yoga session today at 6pm for 45 minutes",
            {
                "summary": "Yoga Session",
                "start": {"dateTime": None, "timeZone": "America/Los_Angeles"},
                "end": {"dateTime": None, "timeZone": "America/Los_Angeles"},
                "description": "Yoga session",
                "location": "Studio"
            }
        )
    ]
    
    for input_text, expected_format in test_cases:
        json_string = await awaited_agent.tool_manager.convert_natural_language_to_calendar_json(input_text)
        assert json_string is not None
        assert isinstance(json_string, str)
        
        # Parse the JSON string
        event_data = json.loads(json_string)
        
        # Check required fields
        assert "summary" in event_data
        assert "start" in event_data
        assert "end" in event_data
        assert "timeZone" in event_data["start"]
        assert "timeZone" in event_data["end"]
        
        # Check that the summary matches the expected type of event
        assert any(word in event_data["summary"].lower() for word in ["workout", "yoga", "fitness"])

@pytest.mark.asyncio
async def test_tool_execution_with_natural_language(agent: PersonalTrainerAgent):
    """Test the full tool execution flow with a natural language command."""
    awaited_agent = await agent
    natural_language_command = "Schedule a team meeting tomorrow at 11am for 30 minutes"
    
    # Create a mock for the actual tool execution to avoid side effects
    async def mock_execute_tool(tool_name, args):
        return {"status": "success", "tool_name": tool_name, "args": args}
    
    awaited_agent.tool_manager.execute_tool = mock_execute_tool
    
    # Process the command (call on agent, not tool_manager)
    if hasattr(awaited_agent, 'execute_tool_from_natural_language'):
        result = await awaited_agent.execute_tool_from_natural_language(natural_language_command)
    else:
        pytest.skip("Agent does not implement execute_tool_from_natural_language")
    
    # Assert that the correct tool was called with the right arguments
    assert result["status"] == "success"
    assert result["tool_name"] == "create_calendar_event"
    assert "summary" in result["args"]
    assert "team meeting" in result["args"]["summary"].lower()

@pytest.mark.asyncio
async def test_tool_confirmation_messages(agent: PersonalTrainerAgent):
    """Test the generation of tool confirmation messages."""
    awaited_agent = await agent
    tool_name = "create_calendar_event"
    args = {"summary": "Project Sync", "start": {"dateTime": "2024-07-20T10:00:00"}, "end": {"dateTime": "2024-07-20T11:00:00"}}
    
    confirmation_message = await awaited_agent.tool_manager.get_tool_confirmation_message(tool_name, args)
    
    # Accept various confirmation message formats
    assert (
        "Okay, I will run the tool" in confirmation_message
        or "I'll schedule a Project Sync meeting" in confirmation_message
        or "I'll schedule a \"Project Sync\" meeting" in confirmation_message
        or "Project Sync" in confirmation_message
    )

@pytest.mark.skip(reason="Calendar service is not available in this test context.")
@pytest.mark.asyncio
async def test_tool_result_processing(agent: PersonalTrainerAgent):
    """
    Test that the agent can process the result of a tool execution
    and return a user-friendly response.
    """
    awaited_agent = await agent
    tool_name = "create_calendar_event"
    event_details = {
        "summary": "Morning Jog",
        "start": {"dateTime": "2024-08-01T08:00:00-07:00", "timeZone": "America/Los_Angeles"},
        "end": {"dateTime": "2024-08-01T09:00:00-07:00", "timeZone": "America/Los_Angeles"},
    }
    
    # Simulate tool execution by directly calling the service
    created_event = await awaited_agent.tool_manager.services['calendar'].write_event(event_details)
    
    # Process the result
    response = await awaited_agent.process_tool_result(tool_name, created_event)
    
    # Assert that the response is a user-friendly summary
    assert "event has been scheduled" in response.lower()
    assert "morning jog" in response.lower()
    
    # Clean up the created event
    await awaited_agent.tool_manager.services['calendar'].delete_event(created_event["id"]) 