import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(as_frame=True)
df = housing.frame

print(df.head())
print("\nDataset information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

print("\nStatistical summary:")
print(df.describe())
from sklearn.model_selection import train_test_split

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel training completed.")
y_pred = model.predict(X_test)

print("\nFirst 5 predictions:")
print(y_pred[:5])

print("\nFirst 5 actual values:")
print(y_test.head().values)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("MAE:", mae)
print("MSE:", mse)
print("R² Score:", r2)
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print("\nFeature coefficients:")
print(coefficients.sort_values(by="Coefficient", ascending=False))
print("\nIntercept:", model.intercept_)
plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red",
    linewidth=2
)

plt.xlabel("Actual House Value")
plt.ylabel("Predicted House Value")
plt.title("Actual vs Predicted House Values")

plt.show()
X_simple = df[["MedInc"]]
y_simple = df["MedHouseVal"]

X_train_simple, X_test_simple, y_train_simple, y_test_simple = train_test_split(
    X_simple, y_simple, test_size=0.2, random_state=42
)

simple_model = LinearRegression()

simple_model.fit(X_train_simple, y_train_simple)

y_pred_simple = simple_model.predict(X_test_simple)

print("\nSimple Linear Regression completed.")
print("Coefficient:", simple_model.coef_[0])
print("Intercept:", simple_model.intercept_)
plt.figure(figsize=(8, 6))

plt.scatter(
    X_test_simple["MedInc"],
    y_test_simple,
    alpha=0.5,
    label="Actual values"
)

plt.plot(
    X_test_simple["MedInc"],
    y_pred_simple,
    color="red",
    linewidth=2,
    label="Regression line"
)

plt.xlabel("Median Income (MedInc)")
plt.ylabel("Median House Value (MedHouseVal)")
plt.title("Simple Linear Regression: Income vs House Value")
plt.legend()

plt.show()