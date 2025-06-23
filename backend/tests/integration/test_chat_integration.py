import pytest
import json
from fastapi.testclient import TestClient
from backend.main import app
from backend.api.routes import initialize_services
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

@pytest.fixture(scope="module")
def client():
    # Run startup events manually
    asyncio.run(initialize_services())
    with TestClient(app) as c:
        yield c

class TestChatIntegration:
    """Integration tests for the chat endpoint."""
    
    @pytest.mark.skipif(not os.getenv("GOOGLE_CLIENT_ID"), reason="Google credentials not provided")
    def test_chat_basic_greeting(self, client):
        """Test basic greeting in chat."""
        request_data = {
            "messages": [
                {"role": "user", "content": "Hello"}
            ]
        }
        
        response = client.post("/api/chat", json=request_data)
        assert response.status_code == 200
        
        response_data = response.json()
        assert "response" in response_data or "responses" in response_data
        
        # Check that we got a response
        if "response" in response_data:
            assert isinstance(response_data["response"], str)
            assert len(response_data["response"]) > 0
        elif "responses" in response_data:
            assert isinstance(response_data["responses"], list)
            assert len(response_data["responses"]) > 0
    
    @pytest.mark.skipif(not os.getenv("GOOGLE_CLIENT_ID"), reason="Google credentials not provided")
    def test_chat_with_conversation_history(self, client):
        """Test chat with conversation history."""
        request_data = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there! How can I help you today?"},
                {"role": "user", "content": "I want to schedule a workout"}
            ]
        }
        
        response = client.post("/api/chat", json=request_data)
        assert response.status_code == 200
        
        response_data = response.json()
        assert "response" in response_data or "responses" in response_data
    
    @pytest.mark.skipif(not os.getenv("GOOGLE_CLIENT_ID"), reason="Google credentials not provided")
    def test_chat_with_tool_calls(self, client):
        """Test chat that should trigger tool calls."""
        request_data = {
            "messages": [
                {"role": "user", "content": "What's on my calendar tomorrow?"}
            ]
        }
        
        response = client.post("/api/chat", json=request_data)
        assert response.status_code == 200
        
        response_data = response.json()
        assert "response" in response_data or "responses" in response_data
    
    @pytest.mark.skipif(not os.getenv("GOOGLE_CLIENT_ID"), reason="Google credentials not provided")
    def test_chat_with_complex_request(self, client):
        """Test chat with complex multi-step request."""
        request_data = {
            "messages": [
                {"role": "user", "content": "Schedule a workout for tomorrow at 2pm and create a task reminder"}
            ]
        }
        
        response = client.post("/api/chat", json=request_data)
        assert response.status_code == 200
        
        response_data = response.json()
        assert "response" in response_data or "responses" in response_data
    
    def test_chat_invalid_request_format(self, client):
        """Test chat with invalid request format."""
        # Missing messages
        request_data = {}
        response = client.post("/api/chat", json=request_data)
        assert response.status_code in [400, 422]  # Bad request or validation error
        
        # Empty messages
        request_data = {"messages": []}
        response = client.post("/api/chat", json=request_data)
        assert response.status_code in [400, 422]
        
        # Invalid message format
        request_data = {"messages": [{"invalid": "format"}]}
        response = client.post("/api/chat", json=request_data)
        assert response.status_code in [400, 422]
    
    def test_chat_malformed_json(self, client):
        """Test chat with malformed JSON."""
        response = client.post("/api/chat", data="invalid json", headers={"Content-Type": "application/json"})
        assert response.status_code == 422  # Unprocessable entity
    
    def test_chat_missing_content_type(self, client):
        """Test chat without proper content type header."""
        request_data = {"messages": [{"role": "user", "content": "Hello"}]}
        response = client.post("/api/chat", data=json.dumps(request_data))
        # FastAPI successfully parses JSON even without Content-Type header
        # The validation happens inside the endpoint logic, not at HTTP level
        assert response.status_code == 200
    
    @pytest.mark.skipif(not os.getenv("GOOGLE_CLIENT_ID"), reason="Google credentials not provided")
    def test_chat_with_system_message(self, client):
        """Test chat with system message."""
        request_data = {
            "messages": [
                {"role": "system", "content": "You are a helpful personal trainer assistant."},
                {"role": "user", "content": "Hello"}
            ]
        }
        
        response = client.post("/api/chat", json=request_data)
        assert response.status_code == 200
        
        response_data = response.json()
        assert "response" in response_data or "responses" in response_data
    
    @pytest.mark.skipif(not os.getenv("GOOGLE_CLIENT_ID"), reason="Google credentials not provided")
    def test_chat_response_consistency(self, client):
        """Test that chat responses are consistent for similar inputs."""
        request_data1 = {"messages": [{"role": "user", "content": "Hello"}]}
        request_data2 = {"messages": [{"role": "user", "content": "Hi"}]}
        
        response1 = client.post("/api/chat", json=request_data1)
        response2 = client.post("/api/chat", json=request_data2)
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        # Both should have valid response structure
        response_data1 = response1.json()
        response_data2 = response2.json()
        
        assert "response" in response_data1 or "responses" in response_data1
        assert "response" in response_data2 or "responses" in response_data2
    
    @pytest.mark.skipif(not os.getenv("GOOGLE_CLIENT_ID"), reason="Google credentials not provided")
    def test_chat_with_long_message(self, client):
        """Test chat with a very long message."""
        long_message = "This is a very long message. " * 100  # Repeat 100 times
        request_data = {
            "messages": [
                {"role": "user", "content": long_message}
            ]
        }
        
        response = client.post("/api/chat", json=request_data)
        assert response.status_code == 200
        
        response_data = response.json()
        assert "response" in response_data or "responses" in response_data
    
    @pytest.mark.skipif(not os.getenv("GOOGLE_CLIENT_ID"), reason="Google credentials not provided")
    def test_chat_with_special_characters(self, client):
        """Test chat with special characters and emojis."""
        special_message = "Hello! 👋 How are you? I want to schedule a workout 🏃‍♂️"
        request_data = {
            "messages": [
                {"role": "user", "content": special_message}
            ]
        }
        
        response = client.post("/api/chat", json=request_data)
        assert response.status_code == 200
        
        response_data = response.json()
        assert "response" in response_data or "responses" in response_data
    
    @pytest.mark.skipif(not os.getenv("GOOGLE_CLIENT_ID"), reason="Google credentials not provided")
    def test_chat_error_handling(self, client):
        """Test chat error handling with problematic inputs."""
        # Test with empty content
        request_data = {
            "messages": [
                {"role": "user", "content": ""}
            ]
        }
        
        response = client.post("/api/chat", json=request_data)
        # Should either return a valid response or a proper error
        assert response.status_code in [200, 400, 422]
        
        # Test with very short content
        request_data = {
            "messages": [
                {"role": "user", "content": "a"}
            ]
        }
        
        response = client.post("/api/chat", json=request_data)
        assert response.status_code in [200, 400, 422]
    
    @pytest.mark.skipif(not os.getenv("GOOGLE_CLIENT_ID"), reason="Google credentials not provided")
    def test_chat_response_time(self, client):
        """Test that chat responses are reasonably fast."""
        import time
        
        request_data = {
            "messages": [
                {"role": "user", "content": "Hello"}
            ]
        }
        
        start_time = time.time()
        response = client.post("/api/chat", json=request_data)
        end_time = time.time()
        
        response_time = end_time - start_time
        
        assert response.status_code == 200
        assert response_time < 30  # Should respond within 30 seconds
    
    @pytest.mark.skipif(not os.getenv("GOOGLE_CLIENT_ID"), reason="Google credentials not provided")
    def test_chat_with_multiple_requests(self, client):
        """Test handling multiple chat requests in sequence."""
        request_data = {
            "messages": [
                {"role": "user", "content": "Hello"}
            ]
        }
        
        # Make multiple requests
        for i in range(3):
            response = client.post("/api/chat", json=request_data)
            assert response.status_code == 200
            
            response_data = response.json()
            assert "response" in response_data or "responses" in response_data 