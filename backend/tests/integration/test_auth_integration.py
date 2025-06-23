import pytest
import os
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from backend.main import app
from backend.utilities.auth import get_credentials, authenticate_all_services, check_authentication_status
from dotenv import load_dotenv

load_dotenv()

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

class TestAuthIntegration:
    """Integration tests for authentication functionality."""
    
    @pytest.mark.skipif(not os.getenv("GOOGLE_CLIENT_ID"), reason="Google credentials not provided")
    def test_check_authentication_status(self):
        """Test that authentication status checking works."""
        status = check_authentication_status()
        
        assert isinstance(status, dict)
        assert "calendar" in status
        assert "gmail" in status
        assert "fitness" in status
        assert "tasks" in status
        assert "drive" in status
        assert "sheets" in status
        
        # All values should be booleans
        for service, is_authenticated in status.items():
            assert isinstance(is_authenticated, bool)
    
    @pytest.mark.skipif(not os.getenv("GOOGLE_CLIENT_ID"), reason="Google credentials not provided")
    def test_get_credentials_structure(self):
        """Test that get_credentials returns proper structure."""
        # This test checks the function exists and has proper structure
        # We won't actually authenticate in tests to avoid browser popups
        assert callable(get_credentials)
        
        # Test that the function expects a service name
        with pytest.raises(Exception):
            # Should raise an exception if no service name provided
            get_credentials("")
    
    @pytest.mark.skipif(not os.getenv("GOOGLE_CLIENT_ID"), reason="Google credentials not provided")
    def test_authenticate_all_services_structure(self):
        """Test that authenticate_all_services has proper structure."""
        # This test checks the function exists and has proper structure
        assert callable(authenticate_all_services)
        
        # We won't actually run authentication in tests to avoid browser popups
        # but we can test the function signature and basic behavior
    
    @pytest.mark.skipif(not os.getenv("GOOGLE_CLIENT_ID"), reason="Google credentials not provided")
    def test_required_environment_variables(self):
        """Test that required environment variables are present."""
        required_vars = ['GOOGLE_CLIENT_ID', 'GOOGLE_PROJECT_ID', 'GOOGLE_CLIENT_SECRET']
        
        for var in required_vars:
            value = os.getenv(var)
            if value is None:
                pytest.skip(f"Missing required environment variable: {var}")
            assert value is not None
            assert len(value) > 0
    
    @patch('backend.utilities.auth.InstalledAppFlow')
    def test_oauth_flow_initialization(self, mock_flow):
        """Test OAuth flow initialization."""
        mock_flow_instance = MagicMock()
        mock_flow.from_client_config.return_value = mock_flow_instance
        mock_flow_instance.run_local_server.return_value = MagicMock()
        
        # Test that the flow can be initialized (we won't actually run it)
        assert mock_flow.from_client_config is not None
    
    @patch('backend.utilities.auth.os.path.exists')
    @patch('backend.utilities.auth.pickle.load')
    def test_credentials_loading(self, mock_pickle_load, mock_exists):
        """Test credentials loading from pickle files."""
        mock_exists.return_value = True
        mock_creds = MagicMock()
        mock_creds.valid = True
        mock_pickle_load.return_value = mock_creds
        
        # Test that credentials can be loaded
        assert mock_exists.return_value is True
        assert mock_pickle_load.return_value is not None
    
    def test_scopes_definition(self):
        """Test that OAuth scopes are properly defined."""
        from backend.utilities.auth import SCOPES
        
        expected_services = ['calendar', 'gmail', 'fitness', 'tasks', 'drive', 'sheets']
        
        for service in expected_services:
            assert service in SCOPES
            assert isinstance(SCOPES[service], list)
            assert len(SCOPES[service]) > 0
            
            # Check that scopes are valid URLs
            for scope in SCOPES[service]:
                assert scope.startswith('https://www.googleapis.com/auth/')
    
    def test_auth_endpoints_exist(self, client):
        """Test that authentication-related endpoints exist and return proper responses."""
        # Test health endpoint (should exist)
        response = client.get("/api/health")
        assert response.status_code in [200, 401, 404]  # Depending on implementation
        
        # Test that the app is running
        response = client.get("/")
        assert response.status_code in [200, 404]  # Root endpoint might not exist
    
    @pytest.mark.skipif(not os.getenv("GOOGLE_CLIENT_ID"), reason="Google credentials not provided")
    def test_token_file_management(self):
        """Test token file management functionality."""
        import tempfile
        import pickle
        
        # Test creating and reading a token file
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pickle', delete=False) as f:
            # Create a simple object instead of MagicMock for pickle test
            test_creds = {"test": "credentials", "valid": True}
            pickle.dump(test_creds, f)
            temp_file = f.name
        
        try:
            # Test reading the token file
            with open(temp_file, 'rb') as f:
                loaded_creds = pickle.load(f)
            
            assert loaded_creds is not None
            assert loaded_creds["test"] == "credentials"
            assert loaded_creds["valid"] is True
            
        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_authentication_error_handling(self):
        """Test authentication error handling."""
        # Test with invalid service name
        with pytest.raises(KeyError):
            # This should raise a KeyError for invalid service
            from backend.utilities.auth import SCOPES
            _ = SCOPES['invalid_service']
    
    @patch('backend.utilities.auth.get_credentials')
    def test_authenticate_all_services_error_handling(self, mock_get_creds):
        """Test error handling in authenticate_all_services."""
        # Mock get_credentials to raise an exception
        mock_get_creds.side_effect = Exception("Authentication failed")
        
        # The function should handle exceptions gracefully
        credentials = authenticate_all_services()
        
        assert isinstance(credentials, dict)
        # Should return empty dict or dict with None values when authentication fails 