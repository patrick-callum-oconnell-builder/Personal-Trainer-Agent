import pytest
import logging
from unittest.mock import AsyncMock, patch, MagicMock
from backend.agent_state_machine import AgentStateMachine
from backend.agent import PersonalTrainerAgent
from langchain_core.messages import HumanMessage, AIMessage
from backend.tests.integration.base_integration_test import BaseIntegrationTest

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.mark.timeout(60)  # Increased timeout to 60 seconds
@pytest.mark.usefixtures("agent")
class TestAgentStateMachine(BaseIntegrationTest):
    @pytest.mark.asyncio
    async def test_decide_next_action_with_tool_call(self, agent: PersonalTrainerAgent):
        """
        Test that decide_next_action correctly identifies a tool call.
        """
        awaited_agent = await agent
        logger.info("Testing decide_next_action with calendar query...")
        history = [HumanMessage(content="What's on my calendar tomorrow?")]
        
        # Create a mock LLM and replace the state machine's LLM
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = AIMessage(content="TOOL_CALL: get_calendar_events tomorrow")
        awaited_agent.state_machine.llm = mock_llm
        
        action = await awaited_agent.state_machine.decide_next_action(history)
        logger.info(f"Action returned: {action}")
        assert action["type"] == "tool_call"
        assert action["tool"] == "get_calendar_events"
        assert "args" in action

    @pytest.mark.asyncio
    async def test_decide_next_action_with_simple_message(self, agent: PersonalTrainerAgent):
        """
        Test that decide_next_action correctly identifies a simple message.
        """
        awaited_agent = await agent
        logger.info("Testing decide_next_action with simple message...")
        history = [HumanMessage(content="Hello there!")]
        # Mock the LLM response to return a simple message
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = AIMessage(content="Hello! How can I help you today?")
        awaited_agent.state_machine.llm = mock_llm
        action = await awaited_agent.state_machine.decide_next_action(history)
        logger.info(f"Action returned: {action}")
        assert action["type"] == "message"
        assert isinstance(action["content"], str)

    @pytest.mark.asyncio
    async def test_process_messages_stream_simple_message(self, agent: PersonalTrainerAgent):
        """
        Test the process_messages_stream with a simple message that doesn't trigger a tool.
        """
        awaited_agent = await agent
        logger.info("Testing process_messages_stream with simple message...")
        history = [HumanMessage(content="Hello!")]
        
        messages = []
        async for message in awaited_agent.state_machine.process_messages_stream(
            messages=history,
            execute_tool_func=awaited_agent.tool_manager.execute_tool,
            get_tool_confirmation_func=awaited_agent.tool_manager.get_tool_confirmation_message,
            summarize_tool_result_func=awaited_agent.tool_manager.summarize_tool_result
        ):
            messages.append(message)
            logger.info(f"Received message: {message}")
        
        assert len(messages) >= 1
        assert isinstance(messages[0], str)

    @pytest.mark.asyncio
    async def test_process_messages_stream_with_tool_call(self, agent: PersonalTrainerAgent):
        """
        Test the process_messages_stream with a message that triggers a tool call.
        """
        awaited_agent = await agent
        logger.info("Testing process_messages_stream with tool call...")
        history = [HumanMessage(content="Schedule a meeting tomorrow at 2pm")]
        
        # Create a mock LLM and replace the state machine's LLM
        mock_llm = AsyncMock()
        # First call: decide action (tool call), Second call: summarize tool result
        mock_llm.ainvoke.side_effect = [
            AIMessage(content="TOOL_CALL: create_calendar_event tomorrow at 2pm"),
            AIMessage(content="I've scheduled your meeting for tomorrow at 2pm.")
        ]
        awaited_agent.state_machine.llm = mock_llm
        
        # Mock the tool execution
        mock_execute = AsyncMock()
        mock_execute.return_value = "Meeting scheduled successfully"
        
        # Mock the tool confirmation
        mock_confirm = AsyncMock()
        mock_confirm.return_value = "I'll schedule a meeting for tomorrow at 2pm."
        
        # Mock the tool result summarization
        mock_summarize = AsyncMock()
        mock_summarize.return_value = "I've scheduled your meeting for tomorrow at 2pm."
        
        messages = []
        async for message in awaited_agent.state_machine.process_messages_stream(
            messages=history,
            execute_tool_func=mock_execute,
            get_tool_confirmation_func=mock_confirm,
            summarize_tool_result_func=mock_summarize
        ):
            messages.append(message)
            logger.info(f"Received message: {message}")
        
        # Should have confirmation and summary messages
        assert len(messages) >= 2
        assert all(isinstance(msg, str) for msg in messages) 