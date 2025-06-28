import unittest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
from typing import Dict, Any
import os
import sys
import json
import asyncio
from langchain_core.messages import HumanMessage

from backend.personal_trainer_agent import PersonalTrainerAgent
from backend.tools.personal_trainer_tool_manager import PersonalTrainerToolManager
from backend.agent_orchestration.auto_tool_manager import AutoToolManager

class TestPersonalTrainerAgent(unittest.IsolatedAsyncioTestCase):
    """Test suite for PersonalTrainerAgent class."""

    async def asyncSetUp(self):
        # Create mock services
        self.mock_calendar = MagicMock()
        self.mock_gmail = MagicMock()
        self.mock_tasks = MagicMock()
        self.mock_drive = MagicMock()
        self.mock_sheets = MagicMock()
        self.mock_maps = MagicMock()
        
        # Create agent with individual services (legacy constructor for unit tests)
        self.agent = PersonalTrainerAgent(
            calendar_service=self.mock_calendar,
            gmail_service=self.mock_gmail,
            tasks_service=self.mock_tasks,
            drive_service=self.mock_drive,
            sheets_service=self.mock_sheets,
            maps_service=self.mock_maps
        )

    async def test_agent_initialization(self):
        """Test that the agent initializes correctly."""
        self.assertIsNotNone(self.agent)
        self.assertIsNotNone(self.agent.state_machine)
        self.assertIsNotNone(self.agent.agent_state)

    async def test_process_message_basic(self):
        """Test basic message processing."""
        message = HumanMessage(content="Hello, how are you?")
        
        # Mock the process_messages_stream method to return an async generator
        async def mock_stream(*args):
            yield "Test response"
        
        # Mock the method to return the generator directly
        self.agent.process_messages_stream = Mock(return_value=mock_stream())
        
        response = await self.agent.process_message(message)
        
        from langchain_core.messages import AIMessage
        self.assertIsInstance(response, AIMessage)
        self.assertEqual(response.content, "Test response")

    async def test_process_message_with_tool_call(self):
        """Test message processing that involves tool calls."""
        message = HumanMessage(content="Schedule a workout for tomorrow at 6pm")
        
        # Mock the process_messages_stream method to simulate a tool call
        mock_response = "I've scheduled your workout for tomorrow at 6pm"
        
        async def mock_stream(*args):
            yield mock_response
        
        # Mock the method to return the generator directly
        self.agent.process_messages_stream = Mock(return_value=mock_stream())
        
        response = await self.agent.process_message(message)
        
        from langchain_core.messages import AIMessage
        self.assertIsInstance(response, AIMessage)
        self.assertEqual(response.content, mock_response)

    async def test_agent_state_management(self):
        """Test that agent state is properly managed."""
        # Directly access fields
        self.assertIsInstance(self.agent.agent_state.messages, list)
        self.assertIsInstance(self.agent.agent_state.status, str)
        # Add a message
        msg = HumanMessage(content="test message")
        self.agent.agent_state.add_message(msg)
        self.assertIn(msg, self.agent.agent_state.messages)

    async def test_error_handling(self):
        """Test error handling in message processing."""
        message = HumanMessage(content="This will cause an error")
        
        # Mock the process_messages_stream to raise an exception
        async def mock_stream_error(*args):
            raise Exception("Test error")
        
        # Mock the method to return the generator that raises an exception
        self.agent.process_messages_stream = Mock(return_value=mock_stream_error())
        
        with self.assertRaises(Exception):
            await self.agent.process_message(message)

    async def test_agent_with_tool_manager(self):
        """Test agent with tool manager integration."""
        # The agent should already have a tool manager from initialization
        self.assertIsNotNone(self.agent.tool_manager)
        self.assertIsInstance(self.agent.tool_manager, PersonalTrainerToolManager)

    async def test_agent_state_persistence(self):
        """Test that agent state persists across operations."""
        # Add a message to the state
        test_message = HumanMessage(content="Test message for persistence")
        self.agent.agent_state.add_message(test_message)
        
        # Check that the state contains the message
        self.assertIn(test_message, self.agent.agent_state.messages)

if __name__ == '__main__':
    unittest.main() 