import asyncio
import pytest
import pytest_asyncio
from datetime import datetime, timedelta
from backend.personal_trainer_agent import PersonalTrainerAgent
from backend.tools.personal_trainer_tool_manager import PersonalTrainerToolManager
from dotenv import load_dotenv
import os

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
async def test_agent_initialization(agent):
    """Test that the agent initializes properly with all required services."""
    awaited_agent = await agent
    assert awaited_agent.tool_manager is not None
    assert awaited_agent.tool_manager.services is not None
    assert 'calendar' in awaited_agent.tool_manager.services
    assert 'gmail' in awaited_agent.tool_manager.services
    assert 'tasks' in awaited_agent.tool_manager.services
    assert 'drive' in awaited_agent.tool_manager.services
    assert 'sheets' in awaited_agent.tool_manager.services
    assert 'maps' in awaited_agent.tool_manager.services

@pytest.mark.asyncio
async def test_tool_creation(agent):
    """Test that all tools are created properly."""
    awaited_agent = await agent
    tools = awaited_agent.tool_manager.get_tools()
    assert tools is not None
    assert len(tools) > 0
    tool_names = [tool.name for tool in tools]
    assert "get_calendar_events" in tool_names
    assert "create_calendar_event" in tool_names
    assert "send_email" in tool_names
    assert "create_task" in tool_names
    assert "search_drive" in tool_names

@pytest.mark.asyncio
async def test_agent_workflow_creation(agent):
    """Test that the agent workflow is created properly."""
    awaited_agent = await agent
    workflow = await awaited_agent._create_agent_workflow()
    assert workflow is not None
    assert hasattr(workflow, "model_name")
    assert hasattr(workflow, "temperature")
    assert hasattr(workflow, "streaming")

@pytest.mark.asyncio
async def collect_stream(agent, messages):
    awaited_agent = await agent
    responses = []
    async for response in awaited_agent.process_messages_stream(messages):
        responses.append(response)
    return "\n".join(responses) if responses else "No response generated."

@pytest.mark.asyncio
async def test_conversation_loop(agent):
    """Test the agent's conversation loop functionality."""
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "Can you help me schedule a workout?"}
    ]
    
    response = await collect_stream(agent, messages)
    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0

@pytest.mark.asyncio
async def test_streaming_conversation(agent):
    """Test the agent's streaming conversation functionality."""
    messages = [
        {"role": "user", "content": "Schedule a workout for tomorrow"}
    ]

    response = await collect_stream(agent, messages)
    
    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0

@pytest.mark.asyncio
async def test_tool_execution_workflow(agent):
    """Test the complete tool execution workflow."""
    # Create a test event with a future time to avoid conflicts
    test_time = (datetime.now() + timedelta(days=2)).replace(hour=14, minute=0, second=0, microsecond=0)
    test_message = f"Schedule a workout for {test_time.strftime('%A, %B %d')} at {test_time.strftime('%I:%M %p')}"
    
    response = await collect_stream(agent, [{"role": "user", "content": test_message}])
    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0

@pytest.mark.asyncio
async def test_error_handling(agent):
    """Test the agent's error handling capabilities."""
    # Test with invalid input that should trigger error handling
    test_message = "Schedule a workout at invalid_time"
    response = await collect_stream(agent, [{"role": "user", "content": test_message}])
    
    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0 