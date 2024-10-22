import pandas as pd

df = pd.read_csv(r"C:\Users\rajes\Downloads\food_recommendation.csv")

# Clean up column names
df.columns = df.columns.str.strip()

# Function to get combined recommendations based on disease severity percentages
def get_combined_recommendations(disease_percentages):
    combined_recommendations = {
        "Fruits(g)": 0,
        "Fruits": set(),
        "Vegetables(g)": 0,
        "Vegetables": set(),
        "Whole Grains(g)": 0,
        "Whole Grains": set(),
        "Leafy Greens(g)": 0,
        "Leafy Greens": set(),
        "Nuts(g)": 0,
        "Nuts": set()
    }

    for condition, percentage in disease_percentages.items():
        if condition in df['Condition'].values:
            index = df[df['Condition'] == condition].index[0]
            combined_recommendations["Fruits(g)"] += df.loc[index, "Fruits(g)"] * (percentage / 100)
            combined_recommendations["Fruits"].update(df.loc[index, "Fruits"].split(', '))
            combined_recommendations["Vegetables(g)"] += df.loc[index, "Vegetables(g)"] * (percentage / 100)
            combined_recommendations["Vegetables"].update(df.loc[index, "Vegetables"].split(', '))
            combined_recommendations["Whole Grains(g)"] += df.loc[index, "Whole Grains(g)"] * (percentage / 100)
            combined_recommendations["Whole Grains"].update(df.loc[index, "Whole Grains"].split(', '))
            combined_recommendations["Leafy Greens(g)"] += df.loc[index, "Leafy Greens(g)"] * (percentage / 100)
            combined_recommendations["Leafy Greens"].update(df.loc[index, "Leafy Greens"].split(', '))
            combined_recommendations["Nuts(g)"] += df.loc[index, "Nuts(g)"] * (percentage / 100)
            combined_recommendations["Nuts"].update(df.loc[index, "Nuts"].split(', '))

    # Convert sets to comma-separated strings
    for key in ["Fruits", "Vegetables", "Whole Grains", "Leafy Greens", "Nuts"]:
        combined_recommendations[key] = ', '.join(combined_recommendations[key])

    return combined_recommendations

# Example input for disease severity percentages
disease_percentages = {'Hypertension': 75}

# Get combined recommendations
combined_recommendations = get_combined_recommendations(disease_percentages)
print(combined_recommendations)