pip install fastapi uvicorn scikit-learn pytest joblib
gedit train.py
python3 train.py
gedit main.py
uvicorn main:app --reload
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"x":abc}'
gedit test_api.py
pip install "fastapi[all]"


pytest

gedit Dockerfile

pip freeze>requirements.txt

Docker build -t test-api-image
Docker run -p 8000:8000 test-api-image
