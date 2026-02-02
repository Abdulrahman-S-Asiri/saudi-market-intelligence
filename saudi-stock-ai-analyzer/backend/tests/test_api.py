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

def test_market_rankings(client):
    # Use quick_scan=True and use_ml=False for speed in tests
    response = client.get("/api/market-rankings?quick_scan=True&use_ml=False")
    # It might return 200 with data or status 'calculating'
    assert response.status_code == 200
    data = response.json()
    # Check for expected keys
    assert "top_bullish" in data or "status" in data
    assert "top_bearish" in data or "status" in data

def test_analyze_stock(client):
    # Test with a known symbol, use train_model=False for speed
    response = client.get("/api/analyze/2222?period=1mo&train_model=False")
    assert response.status_code == 200
    data = response.json()
    assert data["symbol"] == "2222"
    assert "signal" in data

