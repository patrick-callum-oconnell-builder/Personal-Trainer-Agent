import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.api.routes import initialize_services
import os
from dotenv import load_dotenv
import asyncio

# Load environment variables from .env file
load_dotenv()

@pytest.fixture(scope="module")
def client():
    # Run startup events manually
    asyncio.run(initialize_services())
    with TestClient(app) as c:
        yield c

@pytest.mark.skipif(not os.getenv("GOOGLE_CLIENT_ID"), reason="Google credentials not provided")
def test_health_check(client):
    """
    Test the health check endpoint.
    It should return a 200 status code and a 'healthy' status.
    """
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

@pytest.mark.skipif(not os.getenv("GOOGLE_CLIENT_ID"), reason="Google credentials not provided")
def test_chat(client):
    """
    Test the /chat endpoint with a simple message.
    It should return a 200 status code and a response from the agent.
    """
    request_data = {
        "messages": [
            {"role": "user", "content": "Hello"}
        ]
    }
    response = client.post("/api/chat", json=request_data)
    assert response.status_code == 200
    response_data = response.json()
    assert "response" in response_data or "responses" in response_data

@pytest.mark.skipif(not os.getenv("GOOGLE_CLIENT_ID"), reason="Google credentials not provided")
def test_get_calendar_events(client):
    """
    Test the /calendar/events endpoint.
    It should return a 200 status code and a list of events.
    """
    response = client.get("/api/calendar/events")
    assert response.status_code == 200
    response_data = response.json()
    assert "events" in response_data
    assert isinstance(response_data["events"], list)

@pytest.mark.skipif(not os.getenv("GOOGLE_CLIENT_ID"), reason="Google credentials not provided")
def test_get_recent_emails(client):
    """
    Test the /gmail/recent endpoint.
    It should return a 200 status code and a list of emails.
    """
    response = client.get("/api/gmail/recent")
    assert response.status_code == 200
    response_data = response.json()
    assert "emails" in response_data
    assert isinstance(response_data["emails"], list)

@pytest.mark.skipif(not os.getenv("GOOGLE_MAPS_API_KEY"), reason="Google Maps API key not provided")
def test_get_nearby_locations(client):
    """
    Test the /maps/nearby endpoint.
    It should return a 200 status code and a list of locations.
    """
    response = client.get("/api/maps/nearby")
    assert response.status_code == 200
    response_data = response.json()
    assert "locations" in response_data
    assert isinstance(response_data["locations"], list)

@pytest.mark.skipif(not os.getenv("GOOGLE_CLIENT_ID"), reason="Google credentials not provided")
def test_get_fitness_activities(client):
    """
    Test the /fitness/activities endpoint.
    It should return a 200 status code and a list of activities.
    """
    response = client.get("/api/fitness/activities")
    assert response.status_code == 200
    response_data = response.json()
    assert "activities" in response_data

@pytest.mark.skipif(not os.getenv("GOOGLE_CLIENT_ID"), reason="Google credentials not provided")
def test_get_tasks(client):
    """
    Test the /tasks endpoint.
    It should return a 200 status code and a list of tasks.
    """
    response = client.get("/api/tasks")
    assert response.status_code == 200
    response_data = response.json()
    assert "tasks" in response_data
    assert isinstance(response_data["tasks"], list)

@pytest.mark.skipif(not os.getenv("GOOGLE_CLIENT_ID"), reason="Google credentials not provided")
def test_get_knowledge_graph(client):
    """
    Test the /knowledge-graph endpoint.
    It should return a 200 status code and the knowledge graph data.
    """
    # Make a chat call first to populate the agent and knowledge graph
    client.post("/api/chat", json={"messages": [{"role": "user", "content": "Hello"}]})
    
    response = client.get("/api/knowledge-graph")
    assert response.status_code == 200
    response_data = response.json()
    assert "entities" in response_data
    assert "relations" in response_data
    assert isinstance(response_data["entities"], dict)
    assert isinstance(response_data["relations"], list)

@pytest.mark.skipif(not os.getenv("GOOGLE_CLIENT_ID"), reason="Google credentials not provided")
def test_get_state_history(client):
    """
    Test the /state-history endpoint.
    It should return a 200 status code and the state history.
    """
    # Make a chat call first to populate the agent and state history
    client.post("/api/chat", json={"messages": [{"role": "user", "content": "Hello"}]})

    response = client.get("/api/state-history")
    assert response.status_code == 200
    response_data = response.json()
    assert "history" in response_data
    assert isinstance(response_data["history"], list) 