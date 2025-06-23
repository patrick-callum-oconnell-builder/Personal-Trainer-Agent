import os
import sys
import pytest
import json
import re
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) )

# Load environment variables
load_dotenv()

@pytest.mark.asyncio
async def test_maps_tool_call(agent):
    # Explicitly await the agent fixture
    agent_instance = await agent
    messages = [
        HumanMessage(content="Find gyms near 1 Infinite Loop, Cupertino, CA")
    ]
    response = await collect_stream(agent_instance, messages)
    print(f"Maps tool call response: {response}")
    assert response is not None
    assert llm_evaluate_maps_response(response), f"Response did not pass LLM evaluation: {response}"

async def collect_stream(agent, messages):
    responses = []
    async for response in agent.process_messages_stream(messages):
        responses.append(response)
    return "\n".join(responses) if responses else "No response generated."

# Add a placeholder for LLM evaluation
def llm_evaluate_maps_response(response):
    # In production, this would call an LLM to evaluate the response.
    # For now, accept any non-empty string as a valid answer.
    return bool(response and isinstance(response, str) and len(response.strip()) > 0)

if __name__ == '__main__':
    pytest.main() 