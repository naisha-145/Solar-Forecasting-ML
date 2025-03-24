import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score

# Load and clean the dataset
file_path = "Data_solar_on_27-04-2022.xlsx"
df = pd.read_excel(file_path, sheet_name="Solar Data Set", skiprows=2)  # Skipping first 2 rows

df = df.rename(columns=lambda x: x.strip())  # Removing extra spaces from column names

# Selecting features and target variable
features = ['Year', 'Month', 'Day', 'Hour', 'Temperature Units', 'Pressure Units', 'Relative Humidity Units', 'Wind Speed Units']
target = 'Solar Power'  # Assuming this is the target variable

# Drop any rows with missing values
df = df[features + [target]].dropna()

X = df[features]
y = df[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(n_estimators=100),
    'SVR': SVR()
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)  # Train the model
    predictions = model.predict(X_test)  # Get predictions
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    results[name] = {'MAE': mae, 'R-squared': r2}

# Print results
for model, metrics in results.items():
    print(f"{model} - MAE: {metrics['MAE']:.2f}, R-squared: {metrics['R-squared']:.2f}")
