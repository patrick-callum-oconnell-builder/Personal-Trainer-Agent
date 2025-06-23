import asyncio
import pytest
import pytest_asyncio
from datetime import datetime, timedelta
from backend.agent import PersonalTrainerAgent
from backend.google_services import (
    GoogleCalendarService,
    GoogleDriveService,
    GoogleGmailService,
    GoogleMapsService,
    GoogleSheetsService,
    GoogleTasksService,
)
from dotenv import load_dotenv
import os

@pytest_asyncio.fixture
async def agent():
    """Create and initialize an agent for testing."""
    load_dotenv()
    
    # Check for required environment variables
    required_vars = [
        "GOOGLE_MAPS_API_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_CALENDAR_ID"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

    # Initialize services with proper credentials
    calendar_service = GoogleCalendarService()
    await calendar_service.authenticate()  # Use authenticate() instead of async_init()
    
    agent = PersonalTrainerAgent(
        calendar_service=calendar_service,
        gmail_service=GoogleGmailService(),
        tasks_service=GoogleTasksService(),
        drive_service=GoogleDriveService(),
        sheets_service=GoogleSheetsService(),
        maps_service=GoogleMapsService(api_key=os.getenv("GOOGLE_MAPS_API_KEY"))
    )
    await agent.async_init()
    return agent

@pytest.mark.asyncio
async def test_calendar_conflict_detection(agent):
    """Test that calendar conflicts are properly detected."""
    # Create a test event
    test_time = (datetime.now() + timedelta(days=2)).replace(hour=10, minute=0, second=0, microsecond=0)
    test_event = {
        "summary": "Test Event",
        "start": {
            "dateTime": test_time.isoformat(),
            "timeZone": "America/Los_Angeles"
        },
        "end": {
            "dateTime": (test_time + timedelta(hours=1)).isoformat(),
            "timeZone": "America/Los_Angeles"
        }
    }

    # Create the initial event
    await agent.calendar_service.write_event(test_event)

    # Try to create a conflicting event
    conflicting_event = {
        "summary": "Conflicting Event",
        "start": {
            "dateTime": test_time.isoformat(),
            "timeZone": "America/Los_Angeles"
        },
        "end": {
            "dateTime": (test_time + timedelta(hours=1)).isoformat(),
            "timeZone": "America/Los_Angeles"
        }
    }

    # Check for conflicts
    conflicts = await agent.calendar_service.check_for_conflicts(conflicting_event)
    assert len(conflicts) > 0

@pytest.mark.asyncio
async def test_calendar_conflict_resolution_invalid_action(agent):
    """Test handling of invalid conflict resolution actions."""
    test_time = (datetime.now() + timedelta(days=2)).replace(hour=18, minute=0, second=0, microsecond=0)
    test_event = {
        "summary": "Test Event",
        "start": {
            "dateTime": test_time.isoformat(),
            "timeZone": "America/Los_Angeles"
        },
        "end": {
            "dateTime": (test_time + timedelta(hours=1)).isoformat(),
            "timeZone": "America/Los_Angeles"
        }
    }

    # Try to resolve with an invalid action
    response = await agent.tool_manager._resolve_calendar_conflict({
        "event_details": test_event,
        "action": "invalid_action"
    })

    assert response is not None
    assert "error" in response.lower() or "invalid" in response.lower() 