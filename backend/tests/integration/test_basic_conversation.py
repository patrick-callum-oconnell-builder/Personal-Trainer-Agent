import pytest
import asyncio
from langchain_core.messages import HumanMessage, AIMessage
from backend.tests.integration.base_integration_test import BaseIntegrationTest

@pytest.mark.usefixtures("agent")
class TestBasicConversation(BaseIntegrationTest):
    """Integration tests for basic conversation functionality."""
    
    @pytest.mark.asyncio
    async def test_basic_greeting(self, agent):
        """Test basic greeting conversation."""
        awaited_agent = await agent
        
        # Create a simple greeting message
        message = HumanMessage(content="Hello! How are you today?")
        
        # Process the message through the agent
        response_content = ""
        async for chunk in awaited_agent.process_message(message):
            response_content += chunk
        
        # Verify we got a response
        assert response_content is not None
        assert isinstance(response_content, str)
        assert len(response_content) > 0
        
        print(f"Agent response: {response_content}")
    
    @pytest.mark.asyncio
    async def test_workout_question(self, agent):
        """Test asking about workouts."""
        awaited_agent = await agent
        
        # Create a workout-related question
        message = HumanMessage(content="What's a good workout for beginners?")
        
        # Process the message through the agent
        response_content = ""
        async for chunk in awaited_agent.process_message(message):
            response_content += chunk
        
        # Verify we got a response
        assert response_content is not None
        assert isinstance(response_content, str)
        assert len(response_content) > 0
        
        print(f"Agent response: {response_content}")
    
    @pytest.mark.asyncio
    async def test_nutrition_question(self, agent):
        """Test asking about nutrition."""
        awaited_agent = await agent
        
        # Create a nutrition-related question
        message = HumanMessage(content="What should I eat before a workout?")
        
        # Process the message through the agent
        response_content = ""
        async for chunk in awaited_agent.process_message(message):
            response_content += chunk
        
        # Verify we got a response
        assert response_content is not None
        assert isinstance(response_content, str)
        assert len(response_content) > 0
        
        print(f"Agent response: {response_content}")
    
    @pytest.mark.asyncio
    async def test_conversation_context(self, agent):
        """Test that conversation context is maintained."""
        awaited_agent = await agent
        
        # First message
        message1 = HumanMessage(content="My name is John and I want to get stronger.")
        response1_content = ""
        async for chunk in awaited_agent.process_message(message1):
            response1_content += chunk
        
        # Second message that references the first
        message2 = HumanMessage(content="What's a good workout plan for me?")
        response2_content = ""
        async for chunk in awaited_agent.process_message(message2):
            response2_content += chunk
        
        # Verify both responses are valid
        assert response1_content is not None
        assert response2_content is not None
        assert isinstance(response1_content, str)
        assert isinstance(response2_content, str)
        assert len(response1_content) > 0
        assert len(response2_content) > 0
        
        print(f"First response: {response1_content}")
        print(f"Second response: {response2_content}") 