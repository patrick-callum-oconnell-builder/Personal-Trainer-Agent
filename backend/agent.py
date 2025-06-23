# Standard library imports
import asyncio
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, AsyncGenerator
import json

# Third-party imports
from dotenv import load_dotenv
from langchain.schema import BaseMessage
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic.v1 import BaseModel, Field
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langgraph.graph import StateGraph, END

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
from backend.tools.tool_manager import ToolManager
from backend.agent_state import AgentState

load_dotenv()
logger = logging.getLogger(__name__)

# Load configuration
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

AGENT_CONFIG = config['llm']['agent']

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
            model=AGENT_CONFIG['model'],
            temperature=AGENT_CONFIG['temperature'],
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
        
        self.agent_state = AgentState()
        self.graph = self._build_graph()
        
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

    def _build_graph(self):
        # Implementation of _build_graph method
        pass

    async def agent_conversation_loop(self, user_message: str) -> List[str]:
        """Process a user message and return a list of responses."""
        try:
            # Create messages for the LLM
            messages = [
                SystemMessage(content="You are a helpful personal trainer AI assistant. Always respond in clear, natural language. Be concise and direct in stating what action you're about to take."),
                HumanMessage(content=user_message)
            ]
            
            # Get response from LLM
            response = await self.llm.ainvoke(messages)
            
            # Return the response as a list
            return [response.content]
        except Exception as e:
            logger.error(f"Error in agent conversation loop: {str(e)}")
            return [f"I encountered an error: {str(e)}"]
