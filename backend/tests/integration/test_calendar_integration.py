import pytest
import pytest_asyncio
from backend.personal_trainer_agent import PersonalTrainerAgent
from backend.tools.personal_trainer_tool_manager import PersonalTrainerToolManager
from unittest.mock import AsyncMock
from datetime import datetime, timedelta, timezone
import json
import pytz
import asyncio
import re

@pytest_asyncio.fixture
async def tool_manager():
    """Create a tool manager with real calendar service."""
    from backend.api.routes import initialize_services
    services = await initialize_services()
    tool_manager = PersonalTrainerToolManager(services)
    return tool_manager

@pytest_asyncio.fixture
async def agent(tool_manager):
    """Create a fully initialized agent with the tool manager."""
    agent = PersonalTrainerAgent(tool_manager=tool_manager)
    return agent