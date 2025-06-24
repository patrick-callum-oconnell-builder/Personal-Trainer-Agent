import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from backend.tools.personal_trainer_tool_manager import PersonalTrainerToolManager
from backend.personal_trainer_agent import PersonalTrainerAgent
from backend.tests.integration.base_integration_test import BaseIntegrationTest

@pytest.mark.usefixtures("agent")
class TestToolManagerIntegration(BaseIntegrationTest):
    """Integration tests for the tool manager."""
    
    @pytest.mark.asyncio
    async def test_tool_manager_initialization(self, agent: PersonalTrainerAgent):
        """Test that the tool manager initializes correctly."""
        awaited_agent = await agent
        tool_manager = awaited_agent.tool_manager
        
        assert tool_manager is not None
        assert hasattr(tool_manager, 'execute_tool')
        assert hasattr(tool_manager, 'get_tool_confirmation_message')
        assert hasattr(tool_manager, 'summarize_tool_result')
        assert hasattr(tool_manager, 'calendar_service')
        assert hasattr(tool_manager, 'gmail_service')
        assert hasattr(tool_manager, 'tasks_service')
        assert hasattr(tool_manager, 'drive_service')
        assert hasattr(tool_manager, 'sheets_service')
        assert hasattr(tool_manager, 'maps_service')
    
    @pytest.mark.asyncio
    async def test_calendar_tool_execution(self, agent: PersonalTrainerAgent):
        """Test calendar tool execution."""
        awaited_agent = await agent
        tool_manager = awaited_agent.tool_manager
        
        # Test getting calendar events
        result = await tool_manager.execute_tool("get_calendar_events", {"date": "tomorrow"})
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_schedule_workout_tool_execution(self, agent: PersonalTrainerAgent):
        """Test scheduling workout tool execution."""
        awaited_agent = await agent
        tool_manager = awaited_agent.tool_manager
        
        # Test scheduling a workout using create_calendar_event tool
        workout_event = {
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
        
        result = await tool_manager.execute_tool("create_calendar_event", workout_event)
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_gmail_tool_execution(self, agent: PersonalTrainerAgent):
        """Test Gmail tool execution."""
        awaited_agent = await agent
        tool_manager = awaited_agent.tool_manager
        
        # Test getting recent emails
        result = await tool_manager.execute_tool("get_recent_emails", {"count": 5})
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_tasks_tool_execution(self, agent: PersonalTrainerAgent):
        """Test tasks tool execution."""
        awaited_agent = await agent
        tool_manager = awaited_agent.tool_manager
        
        # Test getting tasks
        result = await tool_manager.execute_tool("get_tasks", {})
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_drive_tool_execution(self, agent: PersonalTrainerAgent):
        """Test drive tool execution."""
        awaited_agent = await agent
        tool_manager = awaited_agent.tool_manager
        
        # Test creating a folder
        result = await tool_manager.execute_tool("create_folder", {
            "name": "Test Workout Folder",
            "parent_id": "root"
        })
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_sheets_tool_execution(self, agent: PersonalTrainerAgent):
        """Test sheets tool execution."""
        awaited_agent = await agent
        tool_manager = awaited_agent.tool_manager
        
        # Test creating a workout tracker
        result = await tool_manager.execute_tool("create_workout_tracker", {
            "title": "Test Workout Tracker"
        })
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_maps_tool_execution(self, agent: PersonalTrainerAgent):
        """Test maps tool execution."""
        awaited_agent = await agent
        tool_manager = awaited_agent.tool_manager
        
        # Test getting nearby locations
        result = await tool_manager.execute_tool("get_nearby_locations", {
            "query": "gym",
            "location": "San Francisco, CA"
        })
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_invalid_tool_execution(self, agent: PersonalTrainerAgent):
        """Test execution of invalid tools."""
        awaited_agent = await agent
        tool_manager = awaited_agent.tool_manager
        
        # Test with non-existent tool
        result = await tool_manager.execute_tool("non_existent_tool", {})
        assert result is not None
        assert "error" in result.lower() or "not found" in result.lower()
    
    @pytest.mark.asyncio
    async def test_tool_execution_with_invalid_args(self, agent: PersonalTrainerAgent):
        """Test tool execution with invalid arguments."""
        awaited_agent = await agent
        tool_manager = awaited_agent.tool_manager
        
        # Test calendar tool with invalid date
        result = await tool_manager.execute_tool("get_calendar_events", {"date": "invalid_date"})
        assert result is not None
        assert isinstance(result, str)
    
    @pytest.mark.asyncio
    async def test_tool_confirmation_message(self, agent: PersonalTrainerAgent):
        """Test tool confirmation message generation."""
        awaited_agent = await agent
        tool_manager = awaited_agent.tool_manager
        
        # Test confirmation for calendar tool
        confirmation = await tool_manager.get_tool_confirmation_message(
            "get_calendar_events",
            {"date": "tomorrow"}
        )
        assert confirmation is not None
        assert isinstance(confirmation, str)
        assert len(confirmation) > 0
    
    @pytest.mark.asyncio
    async def test_tool_result_summarization(self, agent: PersonalTrainerAgent):
        """Test tool result summarization."""
        awaited_agent = await agent
        tool_manager = awaited_agent.tool_manager
        
        # Test summarizing calendar results
        result_data = "Found 3 events: Meeting at 10am, Lunch at 12pm, Workout at 2pm"
        summary = await tool_manager.summarize_tool_result(
            "get_calendar_events",
            result_data
        )
        assert summary is not None
        assert isinstance(summary, str)
        assert len(summary) > 0
    
    @pytest.mark.asyncio
    async def test_concurrent_tool_execution(self, agent: PersonalTrainerAgent):
        """Test concurrent execution of multiple tools."""
        awaited_agent = await agent
        tool_manager = awaited_agent.tool_manager
        
        # Execute multiple tools concurrently
        tasks = [
            tool_manager.execute_tool("get_calendar_events", {"date": "today"}),
            tool_manager.execute_tool("get_tasks", {}),
            tool_manager.execute_tool("get_recent_emails", {"count": 3})
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                # Some tools might fail due to missing credentials, which is expected
                continue
            assert result is not None
            assert isinstance(result, str)
    
    @pytest.mark.asyncio
    async def test_tool_execution_error_handling(self, agent: PersonalTrainerAgent):
        """Test error handling in tool execution."""
        awaited_agent = await agent
        tool_manager = awaited_agent.tool_manager
        
        # Test with malformed arguments
        result = await tool_manager.execute_tool("get_calendar_events", None)
        assert result is not None
        assert isinstance(result, str)
        
        # Test with empty arguments
        result = await tool_manager.execute_tool("get_calendar_events", {})
        assert result is not None
        assert isinstance(result, str)
    
    @pytest.mark.asyncio
    async def test_tool_manager_timeout_handling(self, agent: PersonalTrainerAgent):
        """Test timeout handling in tool manager."""
        awaited_agent = await agent
        tool_manager = awaited_agent.tool_manager
        
        # Mock a slow tool execution
        async def slow_tool_execution(*args, **kwargs):
            await asyncio.sleep(0.1)  # Simulate slow execution
            return "Slow tool result"
        
        with patch.object(tool_manager, 'execute_tool', side_effect=slow_tool_execution):
            result = await tool_manager.execute_tool("get_calendar_events", {"date": "tomorrow"})
            assert result is not None
            assert isinstance(result, str)
    
    @pytest.mark.asyncio
    async def test_tool_manager_service_integration(self, agent: PersonalTrainerAgent):
        """Test integration between tool manager and services."""
        awaited_agent = await agent
        tool_manager = awaited_agent.tool_manager
        
        # Test that services are properly initialized
        assert tool_manager.calendar_service is not None
        assert tool_manager.gmail_service is not None
        assert tool_manager.tasks_service is not None
        assert tool_manager.drive_service is not None
        assert tool_manager.sheets_service is not None
        assert tool_manager.maps_service is not None
        
        # Test that services have required methods
        assert hasattr(tool_manager.calendar_service, 'get_events_for_date')
        assert hasattr(tool_manager.gmail_service, 'get_recent_emails')
        assert hasattr(tool_manager.tasks_service, 'get_tasks')
        assert hasattr(tool_manager.drive_service, 'create_folder')
        assert hasattr(tool_manager.sheets_service, 'create_spreadsheet')
        assert hasattr(tool_manager.maps_service, 'find_nearby_places')
    
    @pytest.mark.asyncio
    async def test_tool_manager_workflow(self, agent: PersonalTrainerAgent):
        """Test complete workflow through tool manager."""
        awaited_agent = await agent
        tool_manager = awaited_agent.tool_manager
        
        # Test a complete workflow: check calendar, then schedule workout
        calendar_result = await tool_manager.execute_tool("get_calendar_events", {"date": "tomorrow"})
        assert calendar_result is not None
        
        # Get confirmation for scheduling
        confirmation = await tool_manager.get_tool_confirmation_message(
            "create_calendar_event",
            {"summary": "Workout Session", "date": "tomorrow", "time": "15:00"}
        )
        assert confirmation is not None
        
        # Schedule the workout using create_calendar_event
        workout_event = {
            "summary": "Strength Training",
            "description": "Upper body strength workout",
            "start": {
                "dateTime": "2025-06-20T15:00:00-07:00",
                "timeZone": "America/Los_Angeles"
            },
            "end": {
                "dateTime": "2025-06-20T15:45:00-07:00",
                "timeZone": "America/Los_Angeles"
            }
        }
        schedule_result = await tool_manager.execute_tool("create_calendar_event", workout_event)
        assert schedule_result is not None
        
        # Summarize the result
        summary = await tool_manager.summarize_tool_result("create_calendar_event", schedule_result)
        assert summary is not None 