import pytest
from backend.agent_state_machine import AgentStateMachine
from backend.agent import PersonalTrainerAgent
from langchain_core.messages import HumanMessage
from backend.tests.integration.base_integration_test import BaseIntegrationTest


@pytest.mark.timeout(30)
@pytest.mark.usefixtures("agent")
class TestAgentStateMachine(BaseIntegrationTest):
    @pytest.mark.skip(reason="This test times out due to LLM response delays")
    @pytest.mark.asyncio
    async def test_decide_next_action_with_tool_call(self, agent: PersonalTrainerAgent):
        """
        Test that decide_next_action correctly identifies a tool call.
        """
        awaited_agent = await agent
        history = [HumanMessage(content="What's on my calendar tomorrow?")]
        action = await awaited_agent.state_machine.decide_next_action(history)
        assert action["type"] == "tool_call"
        assert action["tool"] == "get_calendar_events"

    @pytest.mark.skip(reason="This test times out due to LLM response delays")
    @pytest.mark.asyncio
    async def test_decide_next_action_with_simple_message(self, agent: PersonalTrainerAgent):
        """
        Test that decide_next_action correctly identifies a simple message.
        """
        awaited_agent = await agent
        history = [HumanMessage(content="Hello there!")]
        action = await awaited_agent.state_machine.decide_next_action(history)
        assert action["type"] == "message"
        assert isinstance(action["content"], str)

    @pytest.mark.skip(reason="This test is unreliable and times out frequently.")
    @pytest.mark.asyncio
    async def test_process_messages_stream_simple_message(self, agent: PersonalTrainerAgent):
        """
        Test the process_messages_stream with a simple message that doesn't trigger a tool.
        """
        awaited_agent = await agent
        history = [HumanMessage(content="Hello!")]
        
        messages = []
        async for message in awaited_agent.state_machine.process_messages_stream(history):
            messages.append(message)
        
        assert len(messages) == 1
        assert messages[0]["type"] == "message"
        assert isinstance(messages[0]["content"], str)

    @pytest.mark.skip(reason="This test times out due to LLM response delays")
    @pytest.mark.asyncio
    async def test_process_messages_stream_with_tool_call(self, agent: PersonalTrainerAgent):
        """
        Test the process_messages_stream with a message that triggers a tool call.
        """
        awaited_agent = await agent
        history = [HumanMessage(content="Schedule a meeting tomorrow at 2pm")]
        
        messages = []
        async for message in awaited_agent.state_machine.process_messages_stream(history):
            messages.append(message)
        
        # Should have: thinking -> tool_call -> summarize
        assert len(messages) >= 3
        assert messages[0]["type"] == "thinking"
        assert messages[1]["type"] == "tool_call"
        assert messages[-1]["type"] == "summarize" 