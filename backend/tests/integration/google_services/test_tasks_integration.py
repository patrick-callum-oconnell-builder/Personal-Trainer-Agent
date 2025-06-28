import os
import sys
import pytest
from backend.tests.integration.base_integration_test import BaseIntegrationTest

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

class TestTasksIntegration(BaseIntegrationTest):
    @pytest.mark.asyncio
    async def test_create_workout_tasklist(self, agent):
        """Test creating a workout task list."""
        # Explicitly await the agent fixture
        agent_instance = await agent
        try:
            # Create a workout task list using the agent's tasks service
            tasks_service = agent_instance.tool_manager.services['tasks']
            tasklist = await tasks_service.create_workout_tasklist()
            
            # Verify the response structure
            assert tasklist is not None
            assert 'id' in tasklist
            assert 'title' in tasklist
            assert 'workout' in tasklist['title'].lower()
            print(f"Tasks test: Successfully created workout task list with ID {tasklist['id']}")
            
        except Exception as e:
            pytest.fail(f"Failed to create workout task list: {str(e)}")

    @pytest.mark.asyncio
    async def test_add_workout_task(self, agent):
        """Test adding a workout task to the task list."""
        # Explicitly await the agent fixture
        agent_instance = await agent
        try:
            # First create a workout task list
            tasks_service = agent_instance.tool_manager.services['tasks']
            tasklist = await tasks_service.create_workout_tasklist()
            tasklist_id = tasklist['id']
            
            # Add a workout task
            task = await tasks_service.create_task(
                tasklist_id=tasklist_id,
                title="Complete Upper Body Workout",
                notes="Focus on chest and shoulders. 3 sets of each exercise.",
                due="2025-06-25T18:00:00Z"
            )
            
            # Verify the response structure
            assert task is not None
            assert 'id' in task
            assert 'title' in task
            assert task['title'] == "Complete Upper Body Workout"
            print(f"Tasks test: Successfully added workout task with ID {task['id']}")
            
        except Exception as e:
            pytest.fail(f"Failed to add workout task: {str(e)}")

    @pytest.mark.asyncio
    async def test_get_tasks(self, agent):
        """Test getting tasks from the task list."""
        # Explicitly await the agent fixture
        agent_instance = await agent
        try:
            # Get tasks using the agent's tasks service
            tasks_service = agent_instance.tool_manager.services['tasks']
            tasks = await tasks_service.get_tasks()
            
            # Verify the response structure
            assert tasks is not None
            assert isinstance(tasks, list)
            print(f"Tasks test: Successfully fetched {len(tasks)} tasks")
            
            # Log some task details for debugging
            for task in tasks[:3]:  # Show first 3 tasks
                print(f"- {task.get('title', 'No title')} (due: {task.get('due', 'No due date')})")
                
        except Exception as e:
            pytest.fail(f"Failed to get tasks: {str(e)}")

if __name__ == '__main__':
    pytest.main() 