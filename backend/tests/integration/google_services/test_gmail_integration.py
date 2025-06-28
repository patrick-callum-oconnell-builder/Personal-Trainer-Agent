import os
import sys
import pytest
from backend.tests.integration.base_integration_test import BaseIntegrationTest

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

class TestGmailIntegration(BaseIntegrationTest):
    @pytest.mark.asyncio
    async def test_get_recent_emails(self, agent):
        """Test fetching recent emails from Gmail."""
        # Explicitly await the agent fixture
        agent_instance = await agent
        try:
            # Get recent emails using the agent's Gmail service
            gmail_service = agent_instance.tool_manager.services['gmail']
            emails = await gmail_service.get_recent_emails(5)
            
            # Verify the response structure
            assert emails is not None
            assert isinstance(emails, list)
            print(f"Gmail test: Successfully fetched {len(emails)} recent emails")
            
            # Log some email details for debugging
            for email in emails[:3]:  # Show first 3 emails
                print(f"- {email.get('subject', 'No subject')} from {email.get('from', 'No sender')}")
                
        except Exception as e:
            pytest.fail(f"Failed to fetch emails: {str(e)}")

    @pytest.mark.asyncio
    async def test_send_email(self, agent):
        """Test sending an email through Gmail."""
        # Explicitly await the agent fixture
        agent_instance = await agent
        try:
            # Send the email using the agent's Gmail service
            gmail_service = agent_instance.tool_manager.services['gmail']
            result = gmail_service.send_message(
                to="test@example.com",
                subject="Test Workout Progress",
                body="This is a test email for workout progress tracking."
            )
            
            # Verify the response structure
            assert result is not None
            assert 'id' in result
            print(f"Gmail test: Successfully sent email with ID {result['id']}")
            
        except Exception as e:
            if "insufficient authentication scopes" in str(e).lower() or "insufficient permissions" in str(e).lower():
                pytest.skip(f"Gmail send permissions not available: {str(e)}")
            else:
                pytest.fail(f"Failed to send email: {str(e)}")

if __name__ == '__main__':
    pytest.main() 