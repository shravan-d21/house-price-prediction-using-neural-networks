import numpy as np 
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib

# Load dataset
df = pd.read_csv("USA_Housing.csv")  
df.drop(columns=['Address'], inplace=True, errors='ignore')  # Remove non-numeric columns

# Define features and target
X = df.drop(columns=['Price']).values
y = df['Price'].values.reshape(-1, 1)  # Ensure y is 2D for scaling

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
X_scaler = StandardScaler()
X_train_scaled = X_scaler.fit_transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train)  # Scale target variable
y_test_scaled = y_scaler.transform(y_test)

# Save scalers
joblib.dump(X_scaler, "X_scaler.pkl")
joblib.dump(y_scaler, "y_scaler.pkl")

# Build neural network model
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),  # Prevent overfitting
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1)  # Output layer
])

# Compile model with MSE loss and Adam optimizer
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Early stopping callback
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train model
model.fit(X_train_scaled, y_train_scaled,  # Train on scaled y
          validation_data=(X_test_scaled, y_test_scaled), 
          epochs=100, batch_size=16, 
          callbacks=[early_stop], verbose=1)

# Save trained model
model.save("house_price_model.h5")
print("Model trained and saved successfully!")
