from typing import Any, List, Optional
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage

class OrchestratedAgent:
    """
    Base class for orchestrated agents. Handles core agent logic, state management, and orchestration.
    Tool/service logic is injected via dependencies and not hardcoded.
    """
    def __init__(
        self,
        llm: Any,
        tool_manager: Any,
        state_machine_class: Any,
        agent_state_class: Any,
        extract_preference_func: Optional[Any] = None,
        extract_timeframe_func: Optional[Any] = None,
    ):
        self.llm = llm
        self.tool_manager = tool_manager
        self.agent_state = agent_state_class()
        self.state_machine = state_machine_class(
            llm=self.llm,
            tools=self.tool_manager.get_tools(),
            extract_preference_func=extract_preference_func,
            extract_timeframe_func=extract_timeframe_func,
        )

    def _convert_messages_to_base_messages(self, messages: List[Any]) -> List[BaseMessage]:
        """Convert dictionary messages to BaseMessage instances."""
        converted_messages = []
        for msg in messages:
            if isinstance(msg, BaseMessage):
                converted_messages.append(msg)
            elif isinstance(msg, dict):
                role = msg.get('role')
                content = msg.get('content', '')
                if role == 'user':
                    converted_messages.append(HumanMessage(content=content))
                elif role == 'assistant':
                    converted_messages.append(AIMessage(content=content))
                elif role == 'system':
                    converted_messages.append(SystemMessage(content=content))
                else:
                    converted_messages.append(HumanMessage(content=str(msg)))
            else:
                converted_messages.append(HumanMessage(content=str(msg)))
        return converted_messages

    async def process_messages_stream(self, messages: List[BaseMessage]):
        """
        Process messages and return a streaming response with multi-step tool execution.
        """
        # Convert messages to BaseMessage instances if they aren't already
        base_messages = self._convert_messages_to_base_messages(messages)
        
        # Update state with incoming messages
        await self.agent_state.update(
            messages=base_messages,
            status="active"
        )
        
        async for response in self.state_machine.process_messages_stream(
            messages=base_messages,
            execute_tool_func=self.tool_manager.execute_tool,
            get_tool_confirmation_func=self.tool_manager.get_tool_confirmation_message,
            summarize_tool_result_func=self.tool_manager.summarize_tool_result,
            agent_state=self.agent_state
        ):
            # Update state with assistant response
            assistant_message = AIMessage(content=response)
            updated_messages = base_messages + [assistant_message]
            await self.agent_state.update(
                messages=updated_messages,
                status="awaiting_user"
            )
            yield response

    async def extract_preference_llm(self, text: str):
        """Use the LLM to extract a user preference from text. Returns the preference string or None."""
        messages = [
            SystemMessage(content="You are an AI assistant that extracts user preferences from text. Return ONLY the preference (e.g., 'pizza', 'martial arts', 'strength training'), or 'None' if no clear preference is found. Do not include any explanation or extra text."),
            HumanMessage(content=f"Text: {text}")
        ]
        response = await self.llm.ainvoke(messages)
        preference = response.content.strip()
        if preference.lower() == 'none' or not preference:
            return None
        return preference

    async def process_tool_result(self, tool_name: str, result: Any) -> str:
        """Process the result of a tool execution and return a user-friendly response."""
        return await self.tool_manager.summarize_tool_result(tool_name, result) 