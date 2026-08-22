import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import joblib

df = pd.read_csv('D:/Projects/Linear/calories.csv')
df = df.drop(columns=['User_ID'])

df['Gender'] = df['Gender'].map({'male': 1, 'female': 0})

X = df.drop(columns=['Calories'])
y = df['Calories']

feature_names = X.columns.tolist()
print("Features:", feature_names)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
pred_lr = lr.predict(X_test_scaled)

print("\n--- Linear Regression ---")
print(f"R2:   {r2_score(y_test, pred_lr):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, pred_lr)):.4f}")

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
pred_ridge = ridge.predict(X_test_scaled)

print("\n--- Ridge Regression ---")
print(f"R2:   {r2_score(y_test, pred_ridge):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, pred_ridge)):.4f}")

best_model = lr
joblib.dump(best_model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(feature_names, 'features.pkl')

print("\nSaved model.pkl, scaler.pkl, features.pkl")
coefs = pd.Series(best_model.coef_, index=feature_names).sort_values(
    key=abs, ascending=False)
print("\nFeature importance (standardized coefficients):")
print(coefs)
