import pytest
import pytest_asyncio
from backend.personal_trainer_agent import PersonalTrainerAgent
from backend.google_services import GoogleCalendarService
from unittest.mock import AsyncMock
from datetime import datetime, timedelta, timezone
import json
import pytz
import asyncio
import re

@pytest_asyncio.fixture
async def agent():
    """Create a fully initialized agent with a real calendar service."""
    calendar_service = GoogleCalendarService()
    await calendar_service.authenticate()  # Authenticate the service
    agent = PersonalTrainerAgent(
        calendar_service=calendar_service,
        gmail_service=None,
        tasks_service=None,
        drive_service=None,
        sheets_service=None,
        maps_service=None
    )
    return agent