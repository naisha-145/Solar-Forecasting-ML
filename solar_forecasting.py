import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset
df = pd.read_excel("Data_solar_on_27-04-2022.xlsx")

# Strip spaces from column names to avoid mismatch issues
df.columns = df.columns.str.strip()

# Rename incorrectly spelled column
df = df.rename(columns={"Sloar Power": "Solar Power"})

# Print actual column names to verify corrections
print("📌 Column Names After Renaming:")
print(list(df.columns))

# Define features and target
features = ["Temperature Units", "Pressure Units", "Relative Humidity Units", "Wind Speed Units"]
target = "Solar Power"

# Check if all required columns exist in the dataset
missing_features = [col for col in features + [target] if col not in df.columns]

if missing_features:
    print("❌ Missing columns:", missing_features)
else:
    print("✅ All columns are correct!")

# Proceed only if no missing columns
if not missing_features:
    # Selecting required columns
    df = df[features + [target]].dropna()
    print("✅ Dataframe successfully filtered with required columns!")

    # Splitting the data into training and testing sets
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluating the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print("\n📊 Model Evaluation Metrics:")
    print(f"✅ Mean Absolute Error (MAE): {mae:.2f}")
    print(f"✅ Mean Squared Error (MSE): {mse:.2f}")
    print(f"✅ Root Mean Squared Error (RMSE): {rmse:.2f}")
