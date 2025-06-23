import pytest

@pytest.mark.timeout(10)
class BaseIntegrationTest:
    """
    Base class for all integration tests, with a default timeout of 10 seconds.
    All integration test classes should inherit from this class.
    """
    pass 