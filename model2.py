import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib

# Save Ridge model as model_1.pkl
joblib.dump(pipeline_ridge, 'model_2.pkl')

# Load the data
file_name = "cleaned_data_valid_sales_ratio.csv"
data = pd.read_csv(file_name)

# Feature Engineering
data['Log Sale Amount'] = np.log1p(data['Sale Amount'])  # Log transform the Sale Amount

# Define features and target variable
features = ['Town', 'List Year', 'Assessed Value', 'Property Type', 'Residential Type']
X = data[features]
y = data['Log Sale Amount']  # Use log-transformed Sale Amount as the target variable

# Define the column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Assessed Value']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Town', 'Property Type', 'Residential Type'])
    ])

# Create a preprocessing and modeling pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1))  # Reduce the number of trees and enable parallel processing
])

# Use a smaller subset of the data for initial testing
data_sample = data.sample(frac=0.1, random_state=42)  # Use 10% of the data
X_sample = data_sample[features]
y_sample = data_sample['Log Sale Amount']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)

# Train the model
pipeline.fit(X_train, y_train)

# Predict on the test set
y_pred_log = pipeline.predict(X_test)

# Convert predictions back to the original scale
y_pred = np.expm1(y_pred_log)
y_test_original = np.expm1(y_test)

# Evaluate the model
mae = mean_absolute_error(y_test_original, y_pred)
mse = mean_squared_error(y_test_original, y_pred)
r2 = r2_score(y_test_original, y_pred)

print(f"Model Performance:")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# Create a residual plot with a sample of 100 points
residuals = y_test_original - y_pred
sample_size = 100  # Number of points to display in the plot
sample_indices = np.random.choice(len(y_test), sample_size, replace=False)
y_test_sample = y_test_original.iloc[sample_indices]
y_pred_sample = y_pred[sample_indices]
residuals_sample = residuals.iloc[sample_indices]

plt.figure(figsize=(10, 6))
plt.scatter(y_test_sample, residuals_sample, alpha=0.5)
plt.axhline(y=0, color='red', linestyle='--')  # Add a horizontal line at y=0
plt.xlabel('Actual Sale Amount')
plt.ylabel('Residuals')
plt.title('Residual Plot: Actual vs. Predicted Prices')
plt.ylim(-500000, 500000)  # Set the y-axis to start from -500k to 500k
plt.show()

# Function to predict the price for an individual house
def predict_price_for_house(town, list_year, assessed_value, property_type, residential_type):
    # Create a DataFrame for the specific house
    data_house = pd.DataFrame({
        'Town': [town],
        'List Year': [list_year],
        'Assessed Value': [assessed_value],
        'Property Type': [property_type],
        'Residential Type': [residential_type]
    })
    
    # Predict the log-transformed price
    predicted_price_log = pipeline.predict(data_house)
    
    # Convert the prediction back to the original scale
    predicted_price = np.expm1(predicted_price_log)
    return predicted_price[0]

# Example usage for a different house
town = 'Brooklyn'
list_year = 2025
assessed_value = 450000
property_type = 'Single Family'
residential_type = 'Single Family'

predicted_price = predict_price_for_house(town, list_year, assessed_value, property_type, residential_type)
print(f"The predicted sale amount for a house in {town} in {list_year} is: {predicted_price:.2f}")