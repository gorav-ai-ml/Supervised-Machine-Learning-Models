
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = {
    "Study_Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                    2, 3, 4, 5, 6, 7, 8, 9, 1, 10],
    "Marks": [20, 25, 35, 40, 50, 55, 65, 75, 85, 95,
              28, 32, 45, 48, 60, 68, 72, 88, 18, 98]
}

df = pd.DataFrame(data)

X = df[["Study_Hours"]]
y = df["Marks"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=3, random_state=42)
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": round(mae, 2),
        "MSE": round(mse, 2),
        "R2 Score": round(r2, 3)
    })

    print(f"\n{name}")
    print("Actual Marks:    ", list(y_test))
    print("Predicted Marks: ", predictions.round(2).tolist())
    print("MAE:", round(mae, 2))
    print("MSE:", round(mse, 2))
    print("R2 Score:", round(r2, 3))

results_df = pd.DataFrame(results)

print("\nModel Performance Comparison")
print(results_df.to_string(index=False))

plt.scatter(X, y, color="blue", label="Actual Data")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Study Hours vs Marks")
plt.legend()
plt.tight_layout()
plt.savefig("study_hours_vs_marks.png")
plt.show()

results_df.to_csv("model_comparison.csv", index=False)

print("\nProject completed successfully!")
