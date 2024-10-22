import joblib
import pandas as pd

# Load the trained model
model = joblib.load('random_forest_model.pkl')

# Define the features expected by the model
features = ['Age', 'Gender', 'BP', 'Cholesterol', 'Heart Rate', 'Glucose', 'Insulin', 'BMI']

# Define food recommendations based on diseases
food_recommendations = {
    'Heart Disease': {
        'Fruits(g)': 90.0,
        'Fruits': 'Berries, Oranges, Apples, Bananas',
        'Vegetables(g)': 112.5,
        'Vegetables': 'Spinach, Broccoli, Kale, Carrots',
        'Whole Grains(g)': 0.0,
        'Whole Grains': 'N/A',
        'Leafy Greens(g)': 0.0,
        'Leafy Greens': 'N/A',
        'Nuts(g)': 22.5,
        'Nuts': 'Almonds, Walnuts, Pistachios'
    },
    'Hypertension': {
        'Fruits(g)': 150.0,
        'Fruits': 'Bananas, Apples, Citrus Fruits',
        'Vegetables(g)': 225.0,
        'Vegetables': 'Tomatoes, Carrots, Broccoli',
        'Whole Grains(g)': 112.5,
        'Whole Grains': 'Whole Wheat, Oats, Barley',
        'Leafy Greens(g)': 150.0,
        'Leafy Greens': 'Spinach, Kale, Romaine Lettuce',
        'Nuts(g)': 22.5,
        'Nuts': 'Almonds, Walnuts, Flaxseeds'
    }
}

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
    
    # Filter diseases with predicted scores greater than the threshold
    predicted_diseases = {disease: score for disease, score in zip(diseases, predictions[0]) if score > threshold}
    
    # Get food recommendations based on predicted diseases
    combined_food_recommendations = {
        'Fruits(g)': 0.0,
        'Fruits': '',
        'Vegetables(g)': 0.0,
        'Vegetables': '',
        'Whole Grains(g)': 0.0,
        'Whole Grains': '',
        'Leafy Greens(g)': 0.0,
        'Leafy Greens': '',
        'Nuts(g)': 0.0,
        'Nuts': ''
    }

    for disease in predicted_diseases.keys():
        if disease in food_recommendations:
            recommendations = food_recommendations[disease]
            for key, value in recommendations.items():
                if 'g' in key:
                    combined_food_recommendations[key] += value
                else:
                    # Update food items while avoiding duplicates
                    food_items = set(combined_food_recommendations[key].split(', ')) if combined_food_recommendations[key] else set()
                    if value != 'N/A':
                        food_items.update(value.split(', '))
                    combined_food_recommendations[key] = ', '.join(sorted(food_items))  # Sort for consistency

    # Clean up any duplicates in vegetables
    if 'Vegetables' in combined_food_recommendations:
        combined_food_recommendations['Vegetables'] = ', '.join(sorted(set(combined_food_recommendations['Vegetables'].split(', '))))

    return predicted_diseases, combined_food_recommendations

# Example user input
user_input = {
    'Age': 50,
    'Gender': 1,  # 1 for Male, 0 for Female
    'BP': 150,    # Blood Pressure
    'Cholesterol': 350,
    'Heart Rate': 80,
    'Glucose': 100,
    'Insulin': 35,
    'BMI': 20
}

# Predict diseases and get combined food recommendations based on user input
predicted_diseases, combined_food_recommendations = predict_diseases(user_input)

print("Predicted Diseases with Score > 0.5:")
print(predicted_diseases)
print("\nCombined Food Recommendations:")

# Print the combined food recommendations
for key, value in combined_food_recommendations.items():
    print(f"{key}: {value}")
