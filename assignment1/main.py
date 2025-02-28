# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Step 1: Load the dataset
# Assuming the dataset is in a CSV file named 'cpu_performance.csv'
df = pd.read_csv('cpu_performance.csv')

# Step 2: Data Preprocessing
# Separate features and target variable
X = df.drop(columns=['PRP', 'ERP', 'Vendor Name', 'Model Name'])  # Drop non-predictive and target variables
y = df['PRP']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing for numerical features
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
numerical_transformer = StandardScaler()

# Preprocessing for categorical features
categorical_features = X.select_dtypes(include=['object']).columns
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Step 3: Model Development
# Define the model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)  # Output layer for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Create a pipeline that includes preprocessing and the model
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])

# Step 4: Train the Model
pipeline.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Step 5: Evaluate the Model
# Predict on the test set
y_pred = pipeline.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R2): {r2}')

# Step 6: Hyperparameter Tuning
# You can experiment with different hyperparameters such as the number of layers, neurons, learning rate, etc.
# For example, you can try adding more layers or changing the activation functions.

# Example of a more complex model
complex_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])

complex_model.compile(optimizer='adam', loss='mean_squared_error')

complex_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('model', complex_model)])

complex_pipeline.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Evaluate the complex model
y_pred_complex = complex_pipeline.predict(X_test)

mae_complex = mean_absolute_error(y_test, y_pred_complex)
mse_complex = mean_squared_error(y_test, y_pred_complex)
r2_complex = r2_score(y_test, y_pred_complex)

print(f'Complex Model - Mean Absolute Error (MAE): {mae_complex}')
print(f'Complex Model - Mean Squared Error (MSE): {mse_complex}')
print(f'Complex Model - R-squared (R2): {r2_complex}')

# Step 7: Submission
# Save the model and the evaluation metrics for submission
complex_model.save('cpu_performance_model.h5')

# Save the evaluation metrics to a text file
with open('evaluation_metrics.txt', 'w') as f:
    f.write(f'Mean Absolute Error (MAE): {mae}\n')
    f.write(f'Mean Squared Error (MSE): {mse}\n')
    f.write(f'R-squared (R2): {r2}\n')
    f.write(f'Complex Model - Mean Absolute Error (MAE): {mae_complex}\n')
    f.write(f'Complex Model - Mean Squared Error (MSE): {mse_complex}\n')
    f.write(f'Complex Model - R-squared (R2): {r2_complex}\n')