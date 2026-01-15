from sklearn.linear_model import LinearRegression
import joblib
model=LinearRegression()
X=[[1],[2],[3],[4]]
y=[4,5,6,7]
model.fit(X,y)
joblib.dump(model,"model.pkl")
