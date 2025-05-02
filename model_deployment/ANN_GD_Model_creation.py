import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler

# Load your combined DataFrame (ensure df_combined is loaded)
# df_combined = pd.read_pickle('your_pickle_file.pkl')
print("Loaded DataFrame shape:", df_combined.shape)

# Features and target columns
features = ['Age', 'Gender', 'BP', 'Cholesterol', 'Heart Rate', 'Glucose', 'Insulin', 'BMI']
target_columns = ['Heart_Disease', 'Diabetes', 'Stroke', 'Fatty_Liver', 'Metabolic_Syndrome', 'Hypertension']

# Fill missing values
imputer = SimpleImputer(strategy='mean')
df_combined[features] = imputer.fit_transform(df_combined[features])

# Fill missing target values with 0
for target in target_columns:
    df_combined[target] = df_combined.get(target, 0).fillna(0)

# Define X and y
X = df_combined[features]
y = df_combined[target_columns]

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the ANN model
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(y_train.shape[1], activation='sigmoid')  # sigmoid for multi-label binary outputs
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=32, callbacks=[early_stop], verbose=1)

# Evaluate the model
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}, MAE: {mae}, R-squared: {r2}")

# Save the model
model.save('xgboost__best_model.h5')
print("xgboost__best_model.h5")
