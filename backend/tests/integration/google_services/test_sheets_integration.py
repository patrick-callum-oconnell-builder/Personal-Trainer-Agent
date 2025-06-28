import os
import sys
import pytest
from backend.tests.integration.base_integration_test import BaseIntegrationTest

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

class TestSheetsIntegration(BaseIntegrationTest):
    @pytest.mark.asyncio
    async def test_create_workout_tracker(self, agent):
        """Test creating a workout tracker spreadsheet."""
        # Explicitly await the agent fixture
        agent_instance = await agent
        try:
            # Create a workout tracker using the agent's sheets service
            sheets_service = agent_instance.tool_manager.services['sheets']
            spreadsheet = await sheets_service.create_workout_tracker("Workout Tracker")
            
            # Verify the response structure
            assert spreadsheet is not None
            assert 'spreadsheetId' in spreadsheet
            assert 'properties' in spreadsheet
            assert spreadsheet['properties']['title'] == "Workout Tracker"
            print(f"Sheets test: Successfully created workout tracker with ID {spreadsheet['spreadsheetId']}")
            
        except Exception as e:
            pytest.fail(f"Failed to create workout tracker: {str(e)}")

    @pytest.mark.asyncio
    async def test_add_workout_entry(self, agent):
        """Test adding a workout entry to the tracker."""
        # Explicitly await the agent fixture
        agent_instance = await agent
        try:
            # First create a workout tracker
            sheets_service = agent_instance.tool_manager.services['sheets']
            spreadsheet = await sheets_service.create_workout_tracker("Workout Tracker")
            spreadsheet_id = spreadsheet['spreadsheetId']
            
            # Add a workout entry
            result = await sheets_service.add_workout_entry(
                spreadsheet_id=spreadsheet_id,
                date="2025-06-25",
                workout_type="Upper Body",
                duration="60",
                calories="300",
                notes="Good form, felt strong today"
            )
            
            # Verify the response structure
            assert result is not None
            assert 'updates' in result
            assert 'updatedRange' in result['updates']
            print(f"Sheets test: Successfully added workout entry to range {result['updates']['updatedRange']}")
            
        except Exception as e:
            pytest.fail(f"Failed to add workout entry: {str(e)}")

    @pytest.mark.asyncio
    async def test_add_nutrition_entry(self, agent):
        """Test adding a nutrition entry to the tracker."""
        # Explicitly await the agent fixture
        agent_instance = await agent
        try:
            # First create a workout tracker (which includes nutrition tracking)
            sheets_service = agent_instance.tool_manager.services['sheets']
            spreadsheet = await sheets_service.create_workout_tracker("Nutrition Tracker")
            spreadsheet_id = spreadsheet['spreadsheetId']
            
            # Add a nutrition entry
            result = await sheets_service.add_nutrition_entry(
                spreadsheet_id=spreadsheet_id,
                date="2025-06-25",
                meal="Lunch",
                calories="400",
                protein="30",
                carbs="45",
                fat="15",
                notes="Post-workout meal"
            )
            
            # Verify the response structure
            assert result is not None
            assert 'updates' in result
            assert 'updatedRange' in result['updates']
            print(f"Sheets test: Successfully added nutrition entry to range {result['updates']['updatedRange']}")
            
        except Exception as e:
            pytest.fail(f"Failed to add nutrition entry: {str(e)}")

if __name__ == '__main__':
    pytest.main() 