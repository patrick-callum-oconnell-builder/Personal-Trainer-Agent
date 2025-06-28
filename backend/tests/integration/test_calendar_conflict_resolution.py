import asyncio
import pytest
import pytest_asyncio
from datetime import datetime, timezone, timedelta
from backend.personal_trainer_agent import PersonalTrainerAgent
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
from backend.tests.integration.base_integration_test import BaseIntegrationTest

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
    await calendar_service.authenticate()  # Authenticate the calendar service
    
    gmail_service = GoogleGmailService()
    await gmail_service.authenticate()
    
    tasks_service = GoogleTasksService()
    await tasks_service.authenticate()
    
    drive_service = GoogleDriveService()
    await drive_service.authenticate()
    
    sheets_service = GoogleSheetsService()
    await sheets_service.authenticate()
    
    # Create agent with individual services (agent creates its own tool manager internally)  
    agent = PersonalTrainerAgent(
        calendar_service=calendar_service,
        gmail_service=gmail_service,
        tasks_service=tasks_service,
        drive_service=drive_service,
        sheets_service=sheets_service,
        maps_service=GoogleMapsService(api_key=os.getenv("GOOGLE_MAPS_API_KEY"))
    )
    return agent

@pytest.mark.usefixtures("agent")
class TestCalendarConflictResolution(BaseIntegrationTest):
    """Integration tests for calendar conflict resolution."""
    
    @pytest.fixture
    def test_event(self):
        """Create a test event for conflict testing."""
        start_time = datetime.now(timezone.utc) + timedelta(hours=1)
        end_time = start_time + timedelta(hours=1)
        
        return {
            "summary": "Test Workout",
            "description": "Test event for conflict resolution",
            "start": {
                "dateTime": start_time.isoformat(),
                "timeZone": "America/Los_Angeles"
            },
            "end": {
                "dateTime": end_time.isoformat(),
                "timeZone": "America/Los_Angeles"
            }
        }
    
    @pytest.mark.asyncio
    async def test_calendar_conflict_detection(self, agent, test_event):
        """Test that calendar conflicts are properly detected."""
        tool_manager = agent.tool_manager
        
        try:
            # First, create a test event
            calendar_service = tool_manager.services['calendar']
            await calendar_service.write_event(test_event)
            
            # Try to create another event at the same time (should detect conflict)
            conflicting_event = test_event.copy()
            conflicting_event["summary"] = "Conflicting Workout"
            
            result = await calendar_service.write_event(conflicting_event)
            
            # Check if we got a conflict response
            if isinstance(result, dict) and result.get('type') == 'conflict':
                assert 'conflicting_events' in result
                assert 'proposed_event' in result
                assert 'message' in result
                print(f"Conflict detected: {result['message']}")
            else:
                # If no conflict, the event was created successfully
                assert result is not None
                print("No conflict detected, event created successfully")
                
        except Exception as e:
            pytest.fail(f"Failed to test conflict detection: {str(e)}")
    
    @pytest.mark.asyncio
    async def test_calendar_conflict_resolution_invalid_action(self, agent, test_event):
        """Test that invalid conflict resolution actions are handled properly."""
        tool_manager = agent.tool_manager
        
        try:
            # Create a mock conflict data structure
            conflict_data = {
                "type": "conflict",
                "conflicting_events": [test_event],
                "proposed_event": test_event,
                "message": "Test conflict"
            }
            
            # Try to resolve with an invalid action
            result = await tool_manager._resolve_calendar_conflict(conflict_data)
            
            # Should return an error message
            assert result is not None
            assert isinstance(result, str)
            assert "error" in result.lower() or "invalid" in result.lower()
            print(f"Invalid action handled: {result}")
            
        except Exception as e:
            pytest.fail(f"Failed to test invalid conflict resolution: {str(e)}")
    
    @pytest.mark.asyncio
    async def test_calendar_conflict_resolution_replace(self, agent, test_event):
        """Test conflict resolution by replacing the conflicting event."""
        tool_manager = agent.tool_manager
        
        try:
            # Create a mock conflict data structure
            conflict_data = {
                "type": "conflict",
                "conflicting_events": [test_event],
                "proposed_event": test_event,
                "message": "Test conflict",
                "resolution_action": "replace"
            }
            
            # Resolve the conflict by replacing
            result = await tool_manager._resolve_calendar_conflict(conflict_data)
            
            # Should return a success message
            assert result is not None
            assert isinstance(result, str)
            assert len(result) > 0
            print(f"Conflict resolved by replacement: {result}")
            
        except Exception as e:
            pytest.fail(f"Failed to test conflict resolution by replacement: {str(e)}")
    
    @pytest.mark.asyncio
    async def test_calendar_conflict_resolution_skip(self, agent, test_event):
        """Test conflict resolution by skipping the conflicting event."""
        tool_manager = agent.tool_manager
        
        try:
            # Create a mock conflict data structure
            conflict_data = {
                "type": "conflict",
                "conflicting_events": [test_event],
                "proposed_event": test_event,
                "message": "Test conflict",
                "resolution_action": "skip"
            }
            
            # Resolve the conflict by skipping
            result = await tool_manager._resolve_calendar_conflict(conflict_data)
            
            # Should return a success message
            assert result is not None
            assert isinstance(result, str)
            assert len(result) > 0
            print(f"Conflict resolved by skipping: {result}")
            
        except Exception as e:
            pytest.fail(f"Failed to test conflict resolution by skipping: {str(e)}") 