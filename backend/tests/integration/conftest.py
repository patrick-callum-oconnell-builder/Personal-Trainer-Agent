import os
import sys
import pytest
from dotenv import load_dotenv
import asyncio

# Add the backend directory to the Python path
backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(backend_dir)

from backend.personal_trainer_agent import PersonalTrainerAgent
from backend.tools.personal_trainer_tool_manager import PersonalTrainerToolManager

@pytest.fixture(scope="function")
async def google_services():
    """Set up Google services for testing."""
    load_dotenv()
    required_vars = [
        'GOOGLE_CLIENT_ID',
        'GOOGLE_CLIENT_SECRET'
    ]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        pytest.skip(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    # Initialize services using the new architecture
    from backend.api.routes import initialize_services
    services = await initialize_services()
    return services

@pytest.fixture(scope="function")
async def agent(google_services):
    services = await google_services
    # Create agent with individual services (the agent creates its own tool manager internally)
    agent = PersonalTrainerAgent(
        calendar_service=services.get('calendar'),
        gmail_service=services.get('gmail'),
        tasks_service=services.get('tasks'),
        drive_service=services.get('drive'),
        sheets_service=services.get('sheets'),
        maps_service=services.get('maps')
    )
    return agent 