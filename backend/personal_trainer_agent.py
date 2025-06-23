# Standard library imports
import logging
import os
from typing import Any
import json

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
from backend.utilities.time_formatting import extract_timeframe_from_text
from backend.tools.tool_manager import ToolManager

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
        # Initialize the LLM
        llm = ChatOpenAI(
            model=AGENT_CONFIG['model'],
            temperature=AGENT_CONFIG['temperature'],
        )
        # Initialize the ToolManager with all services
        tool_manager = ToolManager(
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
