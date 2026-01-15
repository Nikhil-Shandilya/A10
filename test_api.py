from fastapi.testclient import TestClient
from main import app
client=TestClient(app)
def test_predict():
	result=client.post('/predict',json={"x":6})
	assert result.status_code==200
