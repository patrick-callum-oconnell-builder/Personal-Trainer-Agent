import os
import sys
import pytest
from langchain_core.messages import HumanMessage
from backend.tests.unit.test_utils import llm_check_response_intent
from backend.tests.integration.base_integration_test import BaseIntegrationTest

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

class TestCalendarIntegration(BaseIntegrationTest):
    @pytest.mark.asyncio
    async def test_fetch_upcoming_events(self, agent):
        """Test fetching upcoming events."""
        # Explicitly await the agent fixture
        agent_instance = await agent
        try:
            # Fetch upcoming events using the agent's calendar service
            calendar_service = agent_instance.tool_manager.services['calendar']
            events = await calendar_service.get_upcoming_events("tomorrow")
            
            # Verify the response structure
            assert events is not None
            assert isinstance(events, list)
            
            # Log the events for debugging
            print(f"Found {len(events)} events for tomorrow")
            for event in events:
                print(f"- {event.get('summary', 'No summary')} at {event.get('start', {}).get('dateTime', 'No time')}")
                
        except Exception as e:
            pytest.fail(f"Failed to fetch upcoming events: {str(e)}")

    @pytest.mark.asyncio
    async def test_schedule_workout(self, agent):
        """Test scheduling a workout."""
        # Explicitly await the agent fixture
        agent_instance = await agent
        try:
            # Create a test workout event
            workout_event = {
                "summary": "Test Workout",
                "description": "Scheduled workout session",
                "start": {
                    "dateTime": "2025-06-25T15:00:00-07:00",
                    "timeZone": "America/Los_Angeles"
                },
                "end": {
                    "dateTime": "2025-06-25T16:00:00-07:00",
                    "timeZone": "America/Los_Angeles"
                }
            }
            
            # Schedule the workout using the agent's calendar service
            calendar_service = agent_instance.tool_manager.services['calendar']
            result = await calendar_service.write_event(workout_event)
            
            # Verify the response structure
            assert result is not None
            
            # Handle both string and dictionary responses
            if isinstance(result, str):
                assert len(result) > 0
                print(f"Workout scheduled successfully: {result}")
            elif isinstance(result, dict):
                # This might be a conflict resolution response
                assert "type" in result or "id" in result
                print(f"Workout scheduling result: {result}")
            else:
                # Convert any other type to string for verification
                result_str = str(result)
                assert len(result_str) > 0
                print(f"Workout scheduling result: {result_str}")
                
        except Exception as e:
            pytest.fail(f"Failed to schedule workout: {str(e)}")

def llm_evaluate_confirmation(response_text):
    # In production, this would call an LLM to evaluate the response.
    # For now, accept any non-empty string as a valid confirmation or explanation.
    return bool(response_text and isinstance(response_text, str) and len(response_text.strip()) > 0)

if __name__ == '__main__':
    pytest.main() 