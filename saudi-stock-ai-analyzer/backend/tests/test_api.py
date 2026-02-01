def test_read_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Saudi Stock AI Analyzer API" in response.json()["name"]

def test_health_check(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_get_stocks(client):
    response = client.get("/api/stocks")
    assert response.status_code == 200
    data = response.json()
    assert "stocks" in data
    assert isinstance(data["stocks"], list)
