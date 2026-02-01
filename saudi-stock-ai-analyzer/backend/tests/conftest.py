import pytest
import sys
import os
from fastapi.testclient import TestClient

# Add the backend directory to sys.path so we can import app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app

@pytest.fixture
def client():
    return TestClient(app)
