import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib

# Save Ridge model as model_1.pkl
joblib.dump(pipeline_ridge, 'model_1.pkl')

# Load the data
file_name = "cleaned_data_valid_sales_ratio.csv"
data = pd.read_csv(file_name)

# Feature Engineering
data['Log Sale Amount'] = np.log1p(data['Sale Amount'])  # Log transform the Sale Amount

# Calculate median house prices for each town and year
median_prices = data.groupby(['Town', 'List Year'])['Sale Amount'].median().reset_index()
median_prices.columns = ['Town', 'List Year', 'Median Sale Amount']

# Merge the median prices back to the original data
data = data.merge(median_prices, on=['Town', 'List Year'], how='left')

# Define features and target variable
features = ['Town', 'List Year', 'Assessed Value', 'Property Type', 'Residential Type']
X = data[features]
y = data['Median Sale Amount']

# Define the column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Assessed Value']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Town', 'Property Type', 'Residential Type'])
    ])

# Create a preprocessing and modeling pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', Ridge())
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
pipeline.fit(X_train, y_train)

# Predict on the test set
y_pred = pipeline.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model Performance:")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# Create a scatter plot with a random sample of the test data
sample_size = 100  # Adjust the sample size as needed
sample_indices = np.random.choice(len(y_test), sample_size, replace=False)
y_test_sample = y_test.iloc[sample_indices]
y_pred_sample = y_pred[sample_indices]

plt.figure(figsize=(10, 6))
plt.scatter(y_test_sample, y_pred_sample, alpha=0.5)
plt.xlabel('Actual Median Sale Amount')
plt.ylabel('Predicted Median Sale Amount')
plt.title('Actual vs Predicted Median Sale Amount')
plt.plot([min(y_test_sample), max(y_test_sample)], [min(y_test_sample), max(y_test_sample)], color='red')  # Line of perfect prediction
plt.show()

# List of all towns
all_towns = data['Town'].unique()

# Create a DataFrame for 2024 predictions
data_2024 = pd.DataFrame({
    'Town': all_towns,
    'List Year': [2024] * len(all_towns),
    'Assessed Value': [300000] * len(all_towns),  # Example assessed values, replace with actual values if available
    'Property Type': data['Property Type'].mode()[0],  # Use the most common property type from the training data
    'Residential Type': data['Residential Type'].mode()[0]  # Use the most common residential type from the training data
})

# Predict house prices for all towns in 2024
predicted_prices_2024 = pipeline.predict(data_2024)

# Create a DataFrame to display the results
results_2024 = data_2024[['Town']].copy()
results_2024['Predicted Median Sale Amount'] = predicted_prices_2024

# Save the predictions to a new CSV file
results_2024.to_csv("predicted_median_house_prices_2024.csv", index=False)

print("Predicted median house prices for 2024 have been saved to 'predicted_median_house_prices_2024.csv'.")