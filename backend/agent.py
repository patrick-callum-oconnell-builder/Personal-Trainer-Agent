# Standard library imports
import asyncio
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

# Third-party imports
from dotenv import load_dotenv
from langchain.schema import BaseMessage
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic.v1 import BaseModel, Field

# Local imports
from backend.agent_state_machine import AgentStateMachine
from backend.google_services import (
    GoogleCalendarService,
    GoogleDriveService,
    GoogleGmailService,
    GoogleMapsService,
    GoogleSheetsService,
    GoogleTasksService,
)
from backend.time_formatting import extract_timeframe_from_text
from backend.tool_manager import ToolManager

load_dotenv()
logger = logging.getLogger(__name__)

class FindNearbyWorkoutLocationsInput(BaseModel):
    lat: float = Field(..., description="Latitude of the location")
    lng: float = Field(..., description="Longitude of the location")
    radius: int = Field(5000, description="Search radius in meters (default 5000)")

class PersonalTrainerAgent:
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
        maps_service: Optional[GoogleMapsService] = None
    ):
        """Initialize the personal trainer agent."""
        self.calendar_service = calendar_service
        self.gmail_service = gmail_service
        self.tasks_service = tasks_service
        self.drive_service = drive_service
        self.sheets_service = sheets_service
        self.maps_service = maps_service
        
        # Initialize the LLM
        self.llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.7,
            streaming=True
        )
        
        # Initialize the ToolManager
        self.tool_manager = ToolManager(
            calendar_service=self.calendar_service,
            gmail_service=self.gmail_service,
            tasks_service=self.tasks_service,
            drive_service=self.drive_service,
            sheets_service=self.sheets_service,
            maps_service=self.maps_service,
            llm=self.llm
        )
        
    async def async_init(self):
        """Initialize the agent asynchronously."""
        print("Initializing agent...")
        # Initialize the agent with the custom workflow
        self.agent = await self._create_agent_workflow()
        
        # Initialize the state machine
        self.state_machine = AgentStateMachine(
            llm=self.llm,
            tools=self.tool_manager.get_tools(),
            extract_preference_func=self.extract_preference_llm,
            extract_timeframe_func=extract_timeframe_from_text
        )
        
        print(f"Agent initialized with {len(self.tool_manager.get_tools())} tools:")
        for tool in self.tool_manager.get_tools():
            print(f"- {tool.name}: {tool.description}")
        print("Agent initialization complete.")

    async def _create_agent_workflow(self):
        """Create a simple custom agent that directly uses the LLM and tools."""
        return self.llm

    async def process_tool_result(self, tool_name: str, result: Any) -> str:
        """Process the result of a tool execution and return a user-friendly response."""
        return await self.tool_manager.summarize_tool_result(tool_name, result)

    async def process_messages(self, messages: List[BaseMessage]) -> str:
        """Process a list of messages and return a response as a string."""
        try:
            user_message = None
            for msg in reversed(messages):
                if isinstance(msg, HumanMessage):
                    user_message = msg
                    break
                elif isinstance(msg, dict) and msg.get("role") == "user":
                    user_message = HumanMessage(content=msg.get("content", ""))
                    break
            if not user_message:
                return "I didn't receive any user message to process."
            responses = await self.agent_conversation_loop(user_message.content)
            return "\n".join(responses)
        except Exception as e:
            logger.error(f"Error processing messages: {str(e)}")
            return f"I encountered an error: {str(e)}"

    async def process_messages_stream(self, messages):
        """Process messages and return a streaming response with multi-step tool execution."""
        async for response in self.state_machine.process_messages_stream(
            messages=messages,
            execute_tool_func=self.tool_manager.execute_tool,
            get_tool_confirmation_func=self.tool_manager.get_tool_confirmation_message,
            summarize_tool_result_func=self.tool_manager.summarize_tool_result
        ):
            yield response

    async def extract_preference_llm(self, text: str) -> Optional[str]:
        """Use the LLM to extract a user preference from text. Returns the preference string or None."""
        prompt = (
            "You are an AI assistant that extracts user preferences from text. "
            "Return ONLY the preference (e.g., 'pizza', 'martial arts', 'strength training'), "
            "or 'None' if no clear preference is found. Do not include any explanation or extra text.\n"
            f"Text: {text}"
        )
        messages = [
            SystemMessage(content="You are an AI assistant that extracts user preferences from text. Respond with only the preference or 'None'."),
            HumanMessage(content=prompt)
        ]
        response = await self.llm.ainvoke(messages)
        preference = response.content.strip()
        if preference.lower() == 'none' or not preference:
            return None
        return preference
