import numpy as np
import time
from sklearn.linear_model import LinearRegression
from skl2onnx import to_onnx
import onnxruntime as ort

# 1. Train scikit-learn model
X = np.random.rand(1000, 1)
y = 3 * X.squeeze() + 5

model = LinearRegression()
model.fit(X, y)

# 2. Export model to ONNX
onnx_model = to_onnx(model, X)
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

# 3. Create ONNX Runtime session
sess = ort.InferenceSession("model.onnx")

# 4. Benchmark scikit-learn
t0 = time.time()
for _ in range(10000):
    model.predict(X)
sk_time = time.time() - t0

# 5. Benchmark ONNX Runtime
t0 = time.time()
for _ in range(10000):
    sess.run(None, {"X": X})
onnx_time = time.time() - t0

# 6. Print results
print("scikit-learn time:", sk_time)
print("ONNX Runtime time:", onnx_time)

"""
python3 onn.py
pip install
find ~ -type f -name "activate"
"
