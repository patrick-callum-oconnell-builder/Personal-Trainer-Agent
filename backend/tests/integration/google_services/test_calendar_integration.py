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
            events = await agent_instance.tool_manager.calendar_service.get_upcoming_events()
            assert isinstance(events, list)
            print(f"Calendar test: Successfully fetched {len(events)} upcoming events")
        except Exception as e:
            pytest.fail(f"Failed to fetch upcoming events: {str(e)}")
        
    @pytest.mark.asyncio
    async def test_schedule_workout(self, agent):
        """Test scheduling a workout."""
        # Explicitly await the agent fixture
        agent_instance = await agent
        try:
            # Create a test workout event
            workout = {
                "summary": "Upper Body Workout",
                "description": "Focus on chest and shoulders",
                "start": {
                    "dateTime": "2025-06-20T10:00:00-07:00",
                    "timeZone": "America/Los_Angeles"
                },
                "end": {
                    "dateTime": "2025-06-20T11:00:00-07:00",
                    "timeZone": "America/Los_Angeles"
                }
            }
            
            # Schedule the workout using the agent's calendar service
            result = await agent_instance.tool_manager.calendar_service.write_event(workout)
            assert result is not None
            
            # Check if we got a conflict response
            if 'type' in result and result['type'] == 'conflict':
                # If there's a conflict, try to resolve it by replacing the conflicting event
                result = await agent_instance.tool_manager.calendar_service.resolve_conflict(
                    result['proposed_event'],
                    result['conflicting_events'],
                    'replace'
                )
            
            assert 'id' in result
            print(f"Calendar test: Successfully scheduled workout")
        except Exception as e:
            pytest.fail(f"Failed to schedule workout: {str(e)}")

def llm_evaluate_confirmation(response_text):
    # In production, this would call an LLM to evaluate the response.
    # For now, accept any non-empty string as a valid confirmation or explanation.
    return bool(response_text and isinstance(response_text, str) and len(response_text.strip()) > 0)

if __name__ == '__main__':
    pytest.main() 