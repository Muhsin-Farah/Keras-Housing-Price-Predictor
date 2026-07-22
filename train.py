# train.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("🔄 Loading London housing dataset...")
df = pd.read_csv("data/london_properties_cpi_adjusted.csv")

# 1. Clean Missing Values
df = df.dropna(subset=['distance_to_centre', 'property_age', 'price_adjusted'])

# 2. Extract Raw Numeric Features
numeric_cols = ['distance_to_centre', 'property_age', 'year']
X_numeric = df[numeric_cols].copy()

# 3. One-Hot Encode Text Columns (Convert letters to binary 0 and 1)
categorical_cols = ['tenure_type', 'new_build', 'property_type']
X_categorical = pd.get_dummies(df[categorical_cols], drop_first=False, dtype=float)

# Combine numeric and binary columns into one dataframe
X_combined = pd.concat([X_numeric, X_categorical], axis=1)

# Split features and target price
X = X_combined.to_numpy()
y = df['price_adjusted'].to_numpy()

print(f"📊 Matrix shapes: Features {X.shape} | Targets {y.shape}")
print(f"📝 Total input features going to Keras: {X.shape[1]}")

# 4. Split data into Training and Validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Scale the features (Keeps gradients stable)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Save the scaler so main.py can use it later to scale incoming web requests!
joblib.dump(scaler, "src/data_scaler.joblib")
print("💾 Saved feature scaler to 'src/data_scaler.joblib'")

# 6. Build the Keras Network
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
    layers.BatchNormalization(), 
    layers.Dropout(0.2),         
    
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    
    layers.Dense(32, activation='relu'),
    layers.BatchNormalization(),
    
    layers.Dense(1) # Final continuous price output
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

print("🚀 Training Keras model...")
history = model.fit(
    X_train_scaled, y_train, 
    epochs=40, 
    batch_size=64, 
    validation_data=(X_val_scaled, y_val), 
    verbose=1
)

# 7. Save Loss Curves Plot
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Keras Training Performance')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training_loss.png')
print("📈 Training curve saved to 'training_loss.png'")

# 8. Save the final model artifact
model.save("src/housing_model.keras")
print("💾 Model successfully saved to 'src/housing_model.keras'!")