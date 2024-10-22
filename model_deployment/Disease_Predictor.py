import joblib
import pandas as pd

# Load the trained model
model = joblib.load('random_forest_model.pkl')

# Define the features expected by the model
features = ['Age', 'Gender', 'BP', 'Cholesterol', 'Heart Rate', 'Glucose', 'Insulin', 'BMI']

# Function to predict diseases based on user input
def predict_diseases(user_input, threshold=0.5):
    # Convert user input to a DataFrame
    user_df = pd.DataFrame([user_input])
    
    # Handle missing values (if any)
    user_df.fillna(0, inplace=True)
    
    # Make predictions using the loaded model
    predictions = model.predict(user_df[features])
    
    # Map prediction results to disease names
    diseases = ['Heart Disease', 'Diabetes', 'Stroke', 'Fatty Liver', 'Metabolic Syndrome', 'Hypertension']
    
    # Filter diseases with predicted scores greater than the threshold (e.g., 0.5)
    predicted_diseases = {disease: score for disease, score in zip(diseases, predictions[0]) if score > threshold}
    
    return predicted_diseases

# Example user input
user_input = {
    'Age': 65,
    'Gender': 1,  # 1 for Male, 0 for Female (or as per your dataset encoding)
    'BP': 130,    # Blood Pressure
    'Cholesterol': 250,
    'Heart Rate': 80,
    'Glucose': 100,
    'Insulin': 30,
    'BMI': 25
}

# Predict diseases based on user input
result = predict_diseases(user_input)
print("Predicted Diseases with Score > 0.5:")
print(result)
