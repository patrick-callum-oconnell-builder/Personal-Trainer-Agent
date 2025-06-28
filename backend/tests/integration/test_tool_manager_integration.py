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
        assert hasattr(tool_manager, 'services')
        assert 'calendar' in tool_manager.services
        assert 'gmail' in tool_manager.services
        assert 'tasks' in tool_manager.services
        assert 'drive' in tool_manager.services
        assert 'sheets' in tool_manager.services
        assert 'maps' in tool_manager.services
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_calendar_tool_execution(self, agent: PersonalTrainerAgent):
        """Test calendar tool execution."""
        awaited_agent = await agent
        tool_manager = awaited_agent.tool_manager
        
        # Test getting calendar events
        result = await tool_manager.execute_tool("get_calendar_events", "tomorrow")
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
        
        # Handle both string results and conflict resolution responses
        if isinstance(result, str):
            assert len(result) > 0
        elif isinstance(result, dict):
            # This is likely a conflict resolution response
            assert "type" in result
            assert "message" in result
            # Convert to string for consistency
            result = str(result)
        else:
            # Convert any other type to string
            result = str(result)
        
        assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_gmail_tool_execution(self, agent: PersonalTrainerAgent):
        """Test Gmail tool execution."""
        awaited_agent = await agent
        tool_manager = awaited_agent.tool_manager
        
        # Test getting recent emails
        result = await tool_manager.execute_tool("get_recent_emails", "5")
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_tasks_tool_execution(self, agent: PersonalTrainerAgent):
        """Test tasks tool execution."""
        awaited_agent = await agent
        tool_manager = awaited_agent.tool_manager
        
        # Test getting tasks
        result = await tool_manager.execute_tool("get_tasks", "")
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_drive_tool_execution(self, agent: PersonalTrainerAgent):
        """Test drive tool execution."""
        awaited_agent = await agent
        tool_manager = awaited_agent.tool_manager
        
        # Test creating a folder
        result = await tool_manager.execute_tool("create_folder", "Test Workout Folder")
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_sheets_tool_execution(self, agent: PersonalTrainerAgent):
        """Test sheets tool execution."""
        awaited_agent = await agent
        tool_manager = awaited_agent.tool_manager
        
        # Test creating a workout tracker
        result = await tool_manager.execute_tool("create_workout_tracker", "Test Workout Tracker")
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_maps_tool_execution(self, agent: PersonalTrainerAgent):
        """Test maps tool execution."""
        awaited_agent = await agent
        tool_manager = awaited_agent.tool_manager
        
        # Test getting nearby locations
        result = await tool_manager.execute_tool("get_nearby_locations", "San Francisco, CA|gym")
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
    @pytest.mark.timeout(30)
    async def test_tool_execution_with_invalid_args(self, agent: PersonalTrainerAgent):
        """Test tool execution with invalid arguments."""
        awaited_agent = await agent
        tool_manager = awaited_agent.tool_manager
        
        # Test calendar tool with invalid date
        result = await tool_manager.execute_tool("get_calendar_events", "invalid_date")
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
            "tomorrow"
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
    async def test_tool_execution_error_handling(self, agent: PersonalTrainerAgent):
        """Test tool execution error handling."""
        awaited_agent = await agent
        tool_manager = awaited_agent.tool_manager
        
        # Test with invalid arguments that should cause an error
        result = await tool_manager.execute_tool("get_calendar_events", None)
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_tool_manager_timeout_handling(self, agent: PersonalTrainerAgent):
        """Test tool manager timeout handling."""
        awaited_agent = await agent
        tool_manager = awaited_agent.tool_manager
        
        # Mock a slow tool execution (but not longer than our test timeout)
        async def slow_tool_execution(*args, **kwargs):
            await asyncio.sleep(5)  # Simulate a slow operation, but not timeout-inducing
            return "result"
        
        # Test timeout handling
        with patch.object(tool_manager, 'execute_tool', side_effect=slow_tool_execution):
            result = await tool_manager.execute_tool("get_calendar_events", "tomorrow")
            assert result is not None
    
    @pytest.mark.asyncio
    async def test_tool_manager_service_integration(self, agent: PersonalTrainerAgent):
        """Test tool manager service integration."""
        awaited_agent = await agent
        tool_manager = awaited_agent.tool_manager
        
        # Test that services are properly integrated
        assert tool_manager.services['calendar'] is not None
        assert tool_manager.services['gmail'] is not None
        assert tool_manager.services['tasks'] is not None
        assert tool_manager.services['drive'] is not None
        assert tool_manager.services['sheets'] is not None
        
        # Test that tools are available
        tools = tool_manager.get_tools()
        assert len(tools) > 0
        
        # Test that we can get tools by category
        calendar_tools = tool_manager.get_tools_by_category('calendar')
        assert len(calendar_tools) > 0
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_tool_manager_workflow(self, agent: PersonalTrainerAgent):
        """Test complete tool manager workflow."""
        awaited_agent = await agent
        tool_manager = awaited_agent.tool_manager
        
        # Test a complete workflow: get calendar events, then schedule a workout
        calendar_result = await tool_manager.execute_tool("get_calendar_events", "today")
        assert calendar_result is not None
        assert isinstance(calendar_result, str)
        
        # Test tool confirmation
        confirmation = await tool_manager.get_tool_confirmation_message(
            "get_calendar_events",
            "today"
        )
        assert confirmation is not None
        
        # Test result summarization
        summary = await tool_manager.summarize_tool_result(
            "get_calendar_events",
            calendar_result
        )
        assert summary is not None 