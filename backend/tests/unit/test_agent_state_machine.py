"""
Unit tests for AgentStateMachine

These tests fully mock all dependencies to test only the state transition logic.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import Tool
from backend.agent_orchestration.agent_state_machine import AgentStateMachine
import json
import ast


class TestAgentStateMachine:
    """Test cases for AgentStateMachine state transitions and logic."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        llm = AsyncMock()
        llm.ainvoke = AsyncMock()
        return llm

    @pytest.fixture
    def mock_tools(self):
        """Create mock tools."""
        return [
            Tool(name="get_calendar_events", func=MagicMock(), description="Get calendar events"),
            Tool(name="add_preference_to_kg", func=MagicMock(), description="Add preference to knowledge graph"),
            Tool(name="get_recent_emails", func=MagicMock(), description="Get recent emails")
        ]

    @pytest.fixture
    def mock_extract_preference_func(self):
        """Create a mock preference extraction function."""
        return AsyncMock()

    @pytest.fixture
    def mock_extract_timeframe_func(self):
        """Create a mock timeframe extraction function."""
        return MagicMock()

    @pytest.fixture
    def state_machine(self, mock_llm, mock_tools, mock_extract_preference_func, mock_extract_timeframe_func):
        """Create a state machine instance with mocked dependencies."""
        return AgentStateMachine(
            llm=mock_llm,
            tools=mock_tools,
            extract_preference_func=mock_extract_preference_func,
            extract_timeframe_func=mock_extract_timeframe_func
        )

    @pytest.mark.asyncio
    async def test_init(self, state_machine, mock_llm, mock_tools, mock_extract_preference_func, mock_extract_timeframe_func):
        """Test state machine initialization."""
        assert state_machine.llm == mock_llm
        assert state_machine.tools == mock_tools
        assert state_machine.extract_preference_func == mock_extract_preference_func
        assert state_machine.extract_timeframe_func == mock_extract_timeframe_func

    @pytest.mark.asyncio
    async def test_convert_message_dict_user(self, state_machine):
        """Test converting dict message with user role."""
        msg = {"role": "user", "content": "Hello"}
        result = state_machine._convert_message(msg)
        assert isinstance(result, HumanMessage)
        assert result.content == "Hello"

    @pytest.mark.asyncio
    async def test_convert_message_dict_assistant(self, state_machine):
        """Test converting dict message with assistant role."""
        msg = {"role": "assistant", "content": "Hi there"}
        result = state_machine._convert_message(msg)
        assert isinstance(result, AIMessage)
        assert result.content == "Hi there"

    @pytest.mark.asyncio
    async def test_convert_message_dict_other(self, state_machine):
        """Test converting dict message with other role."""
        msg = {"role": "system", "content": "System message"}
        result = state_machine._convert_message(msg)
        assert isinstance(result, HumanMessage)
        assert result.content == str(msg)

    @pytest.mark.asyncio
    async def test_convert_message_string(self, state_machine):
        """Test converting string message."""
        msg = "Hello world"
        result = state_machine._convert_message(msg)
        assert isinstance(result, HumanMessage)
        assert result.content == "Hello world"

    @pytest.mark.asyncio
    async def test_convert_message_langchain_message(self, state_machine):
        """Test converting LangChain message."""
        msg = HumanMessage(content="Test message")
        result = state_machine._convert_message(msg)
        assert result.content == "Test message"

    @pytest.mark.asyncio
    async def test_decide_next_action_preference_detected(self, state_machine, mock_extract_preference_func):
        """Test decide_next_action when preference is detected."""
        mock_extract_preference_func.return_value = "I like cardio workouts"
        
        result = await state_machine.decide_next_action(["Hello"])
        
        assert result["type"] == "tool_call"
        assert result["tool"] == "add_preference_to_kg"
        assert result["args"] == "I like cardio workouts"
        mock_extract_preference_func.assert_called_once_with("Hello")

    @pytest.mark.asyncio
    async def test_decide_next_action_tool_call_detected(self, state_machine, mock_llm, mock_extract_preference_func, mock_extract_timeframe_func):
        """Test decide_next_action when tool call is detected in LLM response."""
        mock_extract_preference_func.return_value = None
        mock_extract_timeframe_func.return_value = None
        mock_llm.ainvoke.return_value = AIMessage(content="I'll check your calendar.\nTOOL_CALL: get_calendar_events: tomorrow")
        
        result = await state_machine.decide_next_action(["What's on my calendar tomorrow?"])
        
        assert result["type"] == "tool_call"
        assert result["tool"] == "get_calendar_events"
        assert result["args"] == "tomorrow"

    @pytest.mark.asyncio
    async def test_decide_next_action_tool_call_with_timeframe(self, state_machine, mock_llm, mock_extract_preference_func, mock_extract_timeframe_func):
        """Test decide_next_action with timeframe extraction for calendar events."""
        mock_extract_preference_func.return_value = None
        timeframe = {"timeMin": "2025-06-20T00:00:00Z"}
        mock_extract_timeframe_func.return_value = timeframe
        mock_llm.ainvoke.return_value = AIMessage(content="I'll check your calendar.\nTOOL_CALL: get_calendar_events: today")
        
        result = await state_machine.decide_next_action(["What's on my calendar today?"])
        
        assert result["type"] == "tool_call"
        assert result["tool"] == "get_calendar_events"
        # Accept dict or stringified dict (single quotes, possibly double-wrapped)
        if isinstance(result["args"], dict):
            assert result["args"] == timeframe
        else:
            args_str = result["args"]
            # Unwrap and eval until we get a dict
            for _ in range(2):
                if isinstance(args_str, dict):
                    break
                if isinstance(args_str, str):
                    if args_str.startswith('"') and args_str.endswith('"'):
                        args_str = args_str[1:-1]
                    args_str = ast.literal_eval(args_str)
            assert args_str == timeframe
        mock_extract_timeframe_func.assert_called_once_with("What's on my calendar today?")

    @pytest.mark.asyncio
    async def test_decide_next_action_tool_prefix_detected(self, state_machine, mock_llm, mock_extract_preference_func):
        """Test decide_next_action when tool prefix is detected."""
        mock_extract_preference_func.return_value = None
        mock_llm.ainvoke.return_value = AIMessage(content="get_recent_emails: 5")
        
        result = await state_machine.decide_next_action(["Get my recent emails"])
        
        assert result["type"] == "tool_call"
        assert result["tool"] == "get_recent_emails"
        assert result["args"] == "5"

    @pytest.mark.asyncio
    async def test_decide_next_action_message_response(self, state_machine, mock_llm, mock_extract_preference_func):
        """Test decide_next_action when LLM returns a regular message."""
        mock_extract_preference_func.return_value = None
        mock_llm.ainvoke.return_value = AIMessage(content="Hello! How can I help you today?")
        
        result = await state_machine.decide_next_action(["Hello"])
        
        assert result["type"] == "message"
        assert result["content"] == "Hello! How can I help you today?"

    @pytest.mark.asyncio
    async def test_decide_next_action_empty_llm_response(self, state_machine, mock_llm, mock_extract_preference_func):
        """Test decide_next_action when LLM returns empty response."""
        mock_extract_preference_func.return_value = None
        mock_llm.ainvoke.return_value = AIMessage(content="")
        
        with pytest.raises(RuntimeError, match="LLM returned empty response"):
            await state_machine.decide_next_action(["Hello"])

    @pytest.mark.asyncio
    async def test_decide_next_action_history_list_format(self, state_machine, mock_llm, mock_extract_preference_func):
        """Test decide_next_action with list format history."""
        mock_extract_preference_func.return_value = None
        mock_llm.ainvoke.return_value = AIMessage(content="I understand.")
        
        history = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
            HumanMessage(content="How are you?")
        ]
        
        result = await state_machine.decide_next_action(history)
        
        assert result["type"] == "message"
        assert result["content"] == "I understand."

    @pytest.mark.asyncio
    async def test_process_messages_stream_empty_messages(self, state_machine):
        """Test process_messages_stream with empty messages."""
        messages = []
        
        responses = []
        async for response in state_machine.process_messages_stream(
            messages, AsyncMock(), AsyncMock(), AsyncMock()
        ):
            responses.append(response)
        
        assert len(responses) == 1
        assert responses[0] == "I didn't receive any valid messages to process."

    @pytest.mark.asyncio
    async def test_process_messages_stream_no_user_message(self, state_machine):
        """Test process_messages_stream with no user message."""
        messages = [AIMessage(content="Assistant message")]
        mock_execute_tool = AsyncMock()
        mock_get_confirmation = AsyncMock()
        mock_summarize_result = AsyncMock()
        agen = state_machine.process_messages_stream(
            messages, mock_execute_tool, mock_get_confirmation, mock_summarize_result
        )
        response = await agen.__anext__()
        await agen.aclose()
        # Check the first response is a string and matches the error message
        assert isinstance(response, str)
        assert "I need a user message to process." in response
        # Ensure none of the mocks were called
        mock_execute_tool.assert_not_called()
        mock_get_confirmation.assert_not_called()
        mock_summarize_result.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_messages_stream_message_response(self, state_machine, mock_extract_preference_func):
        """Test process_messages_stream when agent decides to send a message."""
        mock_extract_preference_func.return_value = None
        
        with patch.object(state_machine, 'decide_next_action') as mock_decide:
            mock_decide.return_value = {
                "type": "message",
                "content": "Hello! How can I help you?"
            }
            
            messages = [HumanMessage(content="Hello")]
            
            responses = []
            async for response in state_machine.process_messages_stream(
                messages, AsyncMock(), AsyncMock(), AsyncMock()
            ):
                responses.append(response)
            
            assert len(responses) == 1
            assert responses[0] == "Hello! How can I help you?"

    @pytest.mark.asyncio
    async def test_process_messages_stream_tool_call_flow(self, state_machine, mock_extract_preference_func):
        """Test process_messages_stream with tool call flow."""
        mock_extract_preference_func.return_value = None
        
        mock_execute_tool = AsyncMock(return_value="Calendar events retrieved")
        mock_get_confirmation = AsyncMock(return_value="I'll check your calendar")
        mock_summarize_result = AsyncMock(return_value="Here are your calendar events")
        
        with patch.object(state_machine, 'decide_next_action') as mock_decide:
            mock_decide.return_value = {
                "type": "tool_call",
                "tool": "get_calendar_events",
                "args": "tomorrow"
            }
            
            messages = [HumanMessage(content="What's on my calendar tomorrow?")]
            
            responses = []
            async for response in state_machine.process_messages_stream(
                messages, mock_execute_tool, mock_get_confirmation, mock_summarize_result
            ):
                responses.append(response)
            
            assert len(responses) == 2
            assert responses[0] == "I'll check your calendar"
            assert responses[1] == "Here are your calendar events"
            
            mock_execute_tool.assert_called_once_with("get_calendar_events", "tomorrow")
            mock_get_confirmation.assert_called_once_with("get_calendar_events", "tomorrow")
            mock_summarize_result.assert_called_once_with("get_calendar_events", "Calendar events retrieved")

    @pytest.mark.asyncio
    async def test_process_messages_stream_tool_call_empty_summary(self, state_machine, mock_extract_preference_func):
        """Test process_messages_stream when tool summary is empty."""
        mock_extract_preference_func.return_value = None
        
        mock_execute_tool = AsyncMock(return_value="Tool result")
        mock_get_confirmation = AsyncMock(return_value="I'll execute the tool")
        mock_summarize_result = AsyncMock(return_value="")
        
        with patch.object(state_machine, 'decide_next_action') as mock_decide:
            mock_decide.return_value = {
                "type": "tool_call",
                "tool": "get_calendar_events",
                "args": "today"
            }
            
            messages = [HumanMessage(content="Check my calendar")]
            
            responses = []
            async for response in state_machine.process_messages_stream(
                messages, mock_execute_tool, mock_get_confirmation, mock_summarize_result
            ):
                responses.append(response)
            
            # Should get confirmation but then error due to empty summary
            assert len(responses) == 2
            assert responses[0] == "I'll execute the tool"
            assert "Error processing messages" in responses[1]

    @pytest.mark.asyncio
    async def test_process_messages_stream_exception_handling(self, state_machine):
        """Test process_messages_stream exception handling."""
        with patch.object(state_machine, 'decide_next_action', side_effect=Exception("Test error")):
            messages = [HumanMessage(content="Hello")]
            
            responses = []
            async for response in state_machine.process_messages_stream(
                messages, AsyncMock(), AsyncMock(), AsyncMock()
            ):
                responses.append(response)
            
            assert len(responses) == 1
            assert "Error processing messages" in responses[0]

    @pytest.mark.asyncio
    async def test_decide_next_action_exception_handling(self, state_machine, mock_extract_preference_func):
        """Test decide_next_action exception handling."""
        mock_extract_preference_func.side_effect = Exception("Preference extraction error")
        
        with pytest.raises(Exception, match="Preference extraction error"):
            await state_machine.decide_next_action(["Hello"])

    @pytest.mark.asyncio
    async def test_state_machine_complete_workflow(self, state_machine, mock_extract_preference_func):
        """Test complete state machine workflow with multiple state transitions."""
        mock_extract_preference_func.return_value = None
        
        mock_execute_tool = AsyncMock(return_value="Tool executed successfully")
        mock_get_confirmation = AsyncMock(return_value="I'm about to execute the tool")
        mock_summarize_result = AsyncMock(return_value="The tool has been executed successfully")
        
        with patch.object(state_machine, 'decide_next_action') as mock_decide:
            mock_decide.return_value = {
                "type": "tool_call",
                "tool": "get_recent_emails",
                "args": "10"
            }
            
            messages = [HumanMessage(content="Get my recent emails")]
            
            responses = []
            async for response in state_machine.process_messages_stream(
                messages, mock_execute_tool, mock_get_confirmation, mock_summarize_result
            ):
                responses.append(response)
            
            # Verify the complete flow
            assert len(responses) == 2
            assert responses[0] == "I'm about to execute the tool"
            assert responses[1] == "The tool has been executed successfully"
            
            # Verify all functions were called correctly
            mock_execute_tool.assert_called_once_with("get_recent_emails", "10")
            mock_get_confirmation.assert_called_once_with("get_recent_emails", "10")
            mock_summarize_result.assert_called_once_with("get_recent_emails", "Tool executed successfully") 