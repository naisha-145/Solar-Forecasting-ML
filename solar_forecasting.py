import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def load_data(file_path):
    """Load dataset from Excel file."""
    try:
        df = pd.read_excel(file_path)
        return df
    except FileNotFoundError:
        print("❌ The file was not found.")
        return None

def preprocess_data(df):
    """Preprocess data by stripping spaces from column names and renaming."""
    df.columns = df.columns.str.strip()
    df.rename(columns={"Sloar Power": "Solar Power"}, inplace=True, errors='ignore')
    return df

def inspect_data(df):
    """Inspect data types and view the first few rows."""
    print("📊 Data Types:")
    print(df.dtypes)
    print("📝 First Few Rows:")
    print(df.head())

def prepare_features(df):
    """Prepare required features and target."""
    features = ["Temperature Units", "Pressure Units", "Relative Humidity Units", "Wind Speed Units"]
    target = "Solar Power"
    required_columns = features + [target]
    
    # Attempt to convert columns to numeric
    for col in required_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with missing values
    df = df.dropna(subset=required_columns)
    
    # Check if any columns were excluded due to non-numeric data
    excluded_columns = [col for col in required_columns if df[col].dtype not in ['int64', 'float64']]
    if excluded_columns:
        print("❌ Excluded columns due to non-numeric data:", excluded_columns)
        return None, None
    
    print("✅ Dataframe successfully filtered with required columns!")
    
    X = df[features]
    y = df[target]
    return X, y

def train_model(X, y):
    """Train a Random Forest Regressor model."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    print(f"Model MSE: {mse}")
    
    return model

def main():
    file_path = "Data_solar_on_27-04-2022.xlsx"
    df = load_data(file_path)
    
    if df is not None:
        df = preprocess_data(df)
        inspect_data(df)
        X, y = prepare_features(df)
        
        if X is not None and y is not None:
            model = train_model(X, y)

if __name__ == "__main__":
    main()
