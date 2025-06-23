import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from backend.agent_orchestration.agent_state_machine import AgentStateMachine
from backend.personal_trainer_agent import PersonalTrainerAgent
from backend.tests.integration.base_integration_test import BaseIntegrationTest

@pytest.mark.usefixtures("agent")
class TestAgentStateMachineIntegration(BaseIntegrationTest):
    """Integration tests for the agent state machine."""
    
    @pytest.mark.asyncio
    async def test_state_machine_initialization(self, agent: PersonalTrainerAgent):
        """Test that the state machine initializes correctly."""
        awaited_agent = await agent
        state_machine = awaited_agent.state_machine
        
        assert state_machine is not None
        assert hasattr(state_machine, 'decide_next_action')
        assert hasattr(state_machine, 'process_messages_stream')
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_simple_message_decision(self, agent: PersonalTrainerAgent):
        """Test state machine decision for simple messages."""
        awaited_agent = await agent
        state_machine = awaited_agent.state_machine
        
        # Test with a simple greeting
        history = [HumanMessage(content="Hello there!")]
        action = await state_machine.decide_next_action(history)
        
        assert action is not None
        assert "type" in action
        assert action["type"] in ["message", "tool_call"]
        
        if action["type"] == "message":
            assert "content" in action
            assert isinstance(action["content"], str)
        elif action["type"] == "tool_call":
            assert "tool" in action
            assert "args" in action
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_tool_call_decision(self, agent: PersonalTrainerAgent):
        """Test state machine decision for tool calls."""
        awaited_agent = await agent
        state_machine = awaited_agent.state_machine
        
        # Test with a calendar query
        history = [HumanMessage(content="What's on my calendar tomorrow?")]
        action = await state_machine.decide_next_action(history)
        
        assert action is not None
        assert "type" in action
        assert action["type"] in ["message", "tool_call"]
        
        if action["type"] == "tool_call":
            assert "tool" in action
            assert "args" in action
            # Should be a calendar-related tool
            assert any(calendar_tool in action["tool"] for calendar_tool in ["calendar", "get_events"])
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_state_machine_error_recovery(self, agent: PersonalTrainerAgent):
        """Test state machine error recovery."""
        awaited_agent = await agent
        state_machine = awaited_agent.state_machine
        
        # Test with malformed input
        history = [HumanMessage(content="")]
        action = await state_machine.decide_next_action(history)
        
        assert action is not None
        assert "type" in action
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_conversation_history_handling(self, agent: PersonalTrainerAgent):
        """Test state machine with conversation history."""
        awaited_agent = await agent
        state_machine = awaited_agent.state_machine
        
        # Test with conversation history
        history = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there! How can I help you today?"),
            HumanMessage(content="I want to schedule a workout")
        ]
        
        action = await state_machine.decide_next_action(history)
        
        assert action is not None
        assert "type" in action
        assert action["type"] in ["message", "tool_call"]
        
        if action["type"] == "tool_call":
            assert "tool" in action
            assert "args" in action
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_streaming_message_processing(self, agent: PersonalTrainerAgent):
        """Test streaming message processing."""
        awaited_agent = await agent
        state_machine = awaited_agent.state_machine
        
        # Mock tool execution functions
        mock_execute_tool = AsyncMock(return_value="Tool executed successfully")
        mock_get_confirmation = AsyncMock(return_value="Tool confirmed")
        mock_summarize_result = AsyncMock(return_value="Tool result summarized")
        
        history = [HumanMessage(content="Hello")]
        messages = []
        
        async for message in state_machine.process_messages_stream(
            messages=history,
            execute_tool_func=mock_execute_tool,
            get_tool_confirmation_func=mock_get_confirmation,
            summarize_tool_result_func=mock_summarize_result
        ):
            messages.append(message)
        
        assert len(messages) > 0
        assert all(isinstance(msg, str) for msg in messages)
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_streaming_with_tool_calls(self, agent: PersonalTrainerAgent):
        """Test streaming with tool calls."""
        awaited_agent = await agent
        state_machine = awaited_agent.state_machine
        
        # Mock tool execution functions
        mock_execute_tool = AsyncMock(return_value="Calendar events retrieved")
        mock_get_confirmation = AsyncMock(return_value="Calendar tool confirmed")
        mock_summarize_result = AsyncMock(return_value="Here are your calendar events")
        
        history = [HumanMessage(content="Show me my calendar")]
        messages = []
        
        print("[DEBUG] test_streaming_with_tool_calls: Prompt=Show me my calendar")
        start_time = time.time()
        async for message in state_machine.process_messages_stream(
            messages=history,
            execute_tool_func=mock_execute_tool,
            get_tool_confirmation_func=mock_get_confirmation,
            summarize_tool_result_func=mock_summarize_result
        ):
            print(f"[DEBUG] Received message: {message}")
            messages.append(message)
        end_time = time.time()
        print(f"[DEBUG] test_streaming_with_tool_calls: Response time={{end_time - start_time:.2f}}s, Messages={{messages}}")
        
        assert len(messages) > 0
        assert all(isinstance(msg, str) for msg in messages)
    
    @pytest.mark.asyncio
    async def test_state_machine_timeout_handling(self, agent: PersonalTrainerAgent):
        """Test state machine timeout handling."""
        awaited_agent = await agent
        state_machine = awaited_agent.state_machine
        
        # Mock a slow tool execution
        async def slow_execute_tool(*args, **kwargs):
            await asyncio.sleep(0.1)  # Simulate slow execution
            return "Slow tool result"
        
        mock_get_confirmation = AsyncMock(return_value="Tool confirmed")
        mock_summarize_result = AsyncMock(return_value="Tool result summarized")
        
        history = [HumanMessage(content="Test timeout handling")]
        
        # Should not hang indefinitely
        action = await state_machine.decide_next_action(history)
        assert action is not None
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_state_machine_with_system_messages(self, agent: PersonalTrainerAgent):
        """Test state machine with system messages."""
        awaited_agent = await agent
        state_machine = awaited_agent.state_machine
        
        history = [
            SystemMessage(content="You are a helpful personal trainer assistant."),
            HumanMessage(content="Hello")
        ]
        
        action = await state_machine.decide_next_action(history)
        
        assert action is not None
        assert "type" in action
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_state_machine_decision_consistency(self, agent: PersonalTrainerAgent):
        """Test that state machine decisions are consistent for similar inputs."""
        awaited_agent = await agent
        state_machine = awaited_agent.state_machine
        
        # Test with similar inputs
        history1 = [HumanMessage(content="Hello")]
        history2 = [HumanMessage(content="Hi")]
        
        print("[DEBUG] test_state_machine_decision_consistency: Prompt1=Hello, Prompt2=Hi")
        start_time = time.time()
        action1 = await state_machine.decide_next_action(history1)
        print(f"[DEBUG] Action1: {action1}")
        action2 = await state_machine.decide_next_action(history2)
        print(f"[DEBUG] Action2: {action2}")
        end_time = time.time()
        print(f"[DEBUG] test_state_machine_decision_consistency: Response time={{end_time - start_time:.2f}}s")
        
        assert action1 is not None
        assert action2 is not None
        assert "type" in action1
        assert "type" in action2
        
        # Both should be the same type (either message or tool_call)
        assert action1["type"] in ["message", "tool_call"]
        assert action2["type"] in ["message", "tool_call"]
    
    @pytest.mark.asyncio
    async def test_state_machine_with_complex_queries(self, agent: PersonalTrainerAgent):
        """Test state machine with complex queries."""
        awaited_agent = await agent
        state_machine = awaited_agent.state_machine
        
        complex_queries = [
            "Schedule a workout for tomorrow at 2pm and also check my calendar for conflicts",
            "I want to create a workout plan and save it to my drive, then send me an email summary",
            "What's the weather like and can you suggest an indoor workout if it's raining?"
        ]
        
        for query in complex_queries:
            history = [HumanMessage(content=query)]
            action = await state_machine.decide_next_action(history)
            
            assert action is not None
            assert "type" in action
            assert action["type"] in ["message", "tool_call"] 