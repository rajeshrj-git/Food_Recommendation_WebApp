import joblib
import pandas as pd

# Load model
model = joblib.load('model/xgboost_best_model.pkl')

def predict_diseases(user_input, threshold=0.4):
    features = ['Age', 'Gender', 'BP', 'Cholesterol', 'Heart Rate', 'Glucose', 'Insulin', 'BMI']
    user_df = pd.DataFrame([user_input])
    user_df.fillna(0, inplace=True)
    
    predictions = model.predict(user_df[features])
    diseases = ['Heart Disease', 'Diabetes', 'Stroke', 'Fatty Liver', 'Metabolic Syndrome', 'Hypertension']
    
    predicted_diseases = {disease: score for disease, score in zip(diseases, predictions[0]) if score > threshold}
    
    food_recommendations = {
    'Heart Disease': {
        'Fruits(g)': 90.0,
        'Fruits': 'Berries, Oranges, Apples, Bananas',
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
        'Whole Grains(g)': 112.5,
        'Whole Grains': 'Whole Wheat, Oats, Barley',
        'Leafy Greens(g)': 150.0,
        'Leafy Greens': 'Spinach, Kale, Romaine Lettuce',
        'Nuts(g)': 22.5,
        'Nuts': 'Almonds, Walnuts, Flaxseeds'
    },
    'Diabetes': {
        'Fruits(g)': 90.0,
        'Fruits': 'Berries, Apples, Pears, Oranges',
        'Whole Grains(g)': 150.0,
        'Whole Grains': 'Quinoa, Brown Rice, Oats',
        'Leafy Greens(g)': 150.0,
        'Leafy Greens': 'Spinach, Kale, Swiss Chard',
        'Nuts(g)': 22.5,
        'Nuts': 'Almonds, Walnuts, Chia Seeds'
    },
    'Stroke': {
        'Fruits(g)': 120.0,
        'Fruits': 'Citrus Fruits, Apples, Bananas',
        'Whole Grains(g)': 150.0,
        'Whole Grains': 'Oats, Brown Rice, Whole Wheat',
        'Leafy Greens(g)': 150.0,
        'Leafy Greens': 'Kale, Spinach, Collard Greens',
        'Nuts(g)': 22.5,
        'Nuts': 'Walnuts, Almonds, Peanuts'
    },
    'Fatty Liver': {
        'Fruits(g)': 90.0,
        'Fruits': 'Berries, Apples, Grapefruit, Oranges',
        'Whole Grains(g)': 112.5,
        'Whole Grains': 'Oats, Brown Rice, Barley',
        'Leafy Greens(g)': 150.0,
        'Leafy Greens': 'Spinach, Kale, Arugula',
        'Nuts(g)': 22.5,
        'Nuts': 'Almonds, Walnuts, Sunflower Seeds'
    },
    'Metabolic Syndrome': {
        'Fruits(g)': 120.0,
        'Fruits': 'Berries, Citrus Fruits, Apples',
        'Whole Grains(g)': 150.0,
        'Whole Grains': 'Quinoa, Whole Wheat, Oats',
        'Leafy Greens(g)': 150.0,
        'Leafy Greens': 'Kale, Spinach, Romaine Lettuce',
        'Nuts(g)': 22.5,
        'Nuts': 'Almonds, Walnuts, Flaxseeds'
    }
}

    combined_food_recommendations = {'Fruits(g)': 0.0, 'Fruits': '', 'Whole Grains(g)': 0.0, 'Whole Grains': '', 'Leafy Greens(g)': 0.0, 'Leafy Greens': '', 'Nuts(g)': 0.0, 'Nuts': ''}
    
    for disease in predicted_diseases:
        if disease in food_recommendations:
            recommendations = food_recommendations[disease]
            for key, value in recommendations.items():
                if 'g' in key:
                    combined_food_recommendations[key] += value
                else:
                    food_items = set(combined_food_recommendations[key].split(', ')) if combined_food_recommendations[key] else set()
                    food_items.update(value.split(', '))
                    combined_food_recommendations[key] = ', '.join(sorted(food_items))

    return predicted_diseases, combined_food_recommendations
