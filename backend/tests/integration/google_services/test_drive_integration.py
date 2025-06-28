import os
import sys
import pytest
import tempfile
import json
from backend.tests.integration.base_integration_test import BaseIntegrationTest

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

class TestDriveIntegration(BaseIntegrationTest):
    @pytest.mark.asyncio
    async def test_create_folder(self, agent):
        """Test creating a folder in Google Drive."""
        # Explicitly await the agent fixture
        agent_instance = await agent
        try:
            # Create a folder using the agent's drive service
            drive_service = agent_instance.tool_manager.services['drive']
            folder = await drive_service.create_folder("Workout Plans")
            
            # Verify the response structure
            assert folder is not None
            assert 'id' in folder
            print(f"Drive test: Successfully created folder with ID {folder['id']}")
            
        except Exception as e:
            pytest.fail(f"Failed to create workout folder: {str(e)}")

    @pytest.mark.asyncio
    async def test_upload_workout_plan(self, agent):
        """Test uploading a workout plan to Google Drive."""
        # Explicitly await the agent fixture
        agent_instance = await agent
        try:
            # Create a temporary file with workout plan data
            workout_plan = {
                "name": "Upper Body Workout",
                "exercises": [
                    {"name": "Push-ups", "sets": 3, "reps": 15},
                    {"name": "Pull-ups", "sets": 3, "reps": 10},
                    {"name": "Dips", "sets": 3, "reps": 12}
                ],
                "notes": "Focus on form and controlled movements"
            }
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
                json.dump(workout_plan, temp_file)
                temp_file_path = temp_file.name
            
            try:
                # Upload the file using the agent's drive service
                drive_service = agent_instance.tool_manager.services['drive']
                result = drive_service.upload_file(temp_file_path, name="Upper Body Workout Plan.json")
                
                # Verify the response structure
                assert result is not None
                assert 'id' in result
                print(f"Drive test: Successfully uploaded workout plan with ID {result['id']}")
                
            finally:
                # Clean up the temporary file
                os.unlink(temp_file_path)
                
        except Exception as e:
            pytest.fail(f"Failed to upload workout plan: {str(e)}")

    @pytest.mark.asyncio
    async def test_search_files(self, agent):
        """Test searching for files in Google Drive."""
        # Explicitly await the agent fixture
        agent_instance = await agent
        try:
            # Search for files using the agent's drive service
            drive_service = agent_instance.tool_manager.services['drive']
            files = await drive_service.search_files("workout")
            
            # Verify the response structure
            assert files is not None
            assert isinstance(files, list)
            print(f"Drive test: Found {len(files)} files matching 'workout'")
            
            # Log some file details for debugging
            for file in files[:3]:  # Show first 3 files
                print(f"- {file.get('name', 'No name')} ({file.get('mimeType', 'No type')})")
                
        except Exception as e:
            pytest.fail(f"Failed to search files: {str(e)}")

if __name__ == '__main__':
    pytest.main() 