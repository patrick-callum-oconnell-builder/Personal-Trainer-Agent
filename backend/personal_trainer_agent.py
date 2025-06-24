# Standard library imports
import json
import logging
import os
from typing import Any

# Third-party imports
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Local imports
from backend.agent_orchestration import OrchestratedAgent, AgentStateMachine, AgentState
from backend.google_services import (
    GoogleCalendarService,
    GoogleDriveService,
    GoogleGmailService,
    GoogleMapsService,
    GoogleSheetsService,
    GoogleTasksService,
)
from backend.tools.personal_trainer_tool_manager import PersonalTrainerToolManager
from backend.utilities.time_formatting import extract_timeframe_from_text

load_dotenv()
logger = logging.getLogger(__name__)

# Load configuration
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

AGENT_CONFIG = config['llm']['agent']

class PersonalTrainerAgent(OrchestratedAgent):
    """
    An AI-powered personal trainer agent that integrates with various Google services
    to provide personalized workout recommendations and tracking.
    """
    def __init__(
        self,
        calendar_service: GoogleCalendarService,
        gmail_service: GoogleGmailService,
        tasks_service: GoogleTasksService,
        drive_service: GoogleDriveService,
        sheets_service: GoogleSheetsService,
        maps_service: GoogleMapsService = None
    ):
        # Initialize the LLM with timeout configuration
        llm = ChatOpenAI(
            model=AGENT_CONFIG['model'],
            temperature=AGENT_CONFIG['temperature'],
            timeout=30,  # 30 second timeout
            max_retries=2,  # Retry up to 2 times
            request_timeout=30,  # Request timeout
        )
        # Initialize the ToolManager with all services
        tool_manager = PersonalTrainerToolManager(
            calendar_service=calendar_service,
            gmail_service=gmail_service,
            tasks_service=tasks_service,
            drive_service=drive_service,
            sheets_service=sheets_service,
            maps_service=maps_service,
            llm=llm
        )
        super().__init__(
            llm=llm,
            tool_manager=tool_manager,
            state_machine_class=AgentStateMachine,
            agent_state_class=AgentState,
            extract_preference_func=super().extract_preference_llm,
            extract_timeframe_func=extract_timeframe_from_text
        )
