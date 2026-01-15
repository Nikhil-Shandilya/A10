from sklearn.linear_model import LinearRegression
import joblib
model=LinearRegression()
X=[[1],[2],[3],[4]]
y=[4,5,6,7]
model.fit(X,y)
joblib.dump(model,"model.pkl")

"""
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



find ~ -type f -name "activate"

find ~ -type f -name "activate"


/home/nick/myenv/bin/activate
cmd: /home/nick/myenv/bin/python clean.py

~/.venvs/myenv
cmd: $HOME/.venvs/myenv/bin/python clean.py

project/
  .venv/
  cmd: .venv/bin/python clean.py

which python
/home/nick/myenv/bin/python

python -c "import sys; print(sys.executable)"
/home/nick/myenv/bin/python
"""
