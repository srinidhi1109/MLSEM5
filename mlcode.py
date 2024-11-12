import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import streamlit as st

# Load and preprocess the data
file_path = 'crop_yield_data_sheet.xlsx'  # Ensure this file is in the same directory as your script
data = pd.read_excel(file_path)

# Clean data
features = ['Rain Fall (mm)', 'Fertilizer', 'Temperatue', 'Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)']
data[features] = data[features].apply(pd.to_numeric, errors='coerce')
data = data.dropna()

# Prepare features and target
X = data[features]
y = data['Yeild (Q/acre)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
models = {
    'SVR': SVR(kernel='rbf'),
    'Random Forest': RandomForestRegressor(n_estimators=100),
    'Decision Tree': DecisionTreeRegressor(),
    'KNN': KNeighborsRegressor(n_neighbors=5)
}

# Function to train models and return their predictions and evaluation metrics
def train_models():
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        cv = np.std(y_pred) / np.mean(y_pred) * 100  # Percentage CV

        results[name] = {
            'model': model,  # Store the trained model for later use
            'predictions': y_pred,
            'metrics': {
                'MSE': mse,
                'R²': r2,
                'MAE': mae,
                'RMSE': rmse,
                'CV': cv
            }
        }
    return results

# Train models and store results at the start
trained_results = train_models()

# Define thresholds for parameters
high_yield_threshold = 9.0
low_fertilizer_threshold = 50.0
high_rain_fall_threshold = 1200.0
optimal_temperature_range = (20, 30)
low_nitrogen_threshold = 30.0
high_nitrogen_threshold = 80.0
low_phosphorus_threshold = 15.0
high_phosphorus_threshold = 30.0
low_potassium_threshold = 10.0
high_potassium_threshold = 25.0

# Prediction function
def predict_crop_yield(rain_fall, fertilizer, temperature, nitrogen, phosphorus, potassium):
    input_data = pd.DataFrame([[rain_fall, fertilizer, temperature, nitrogen, phosphorus, potassium]], columns=features)
    predicted_yield = trained_results['SVR']['model'].predict(input_data)  # Use the trained SVR model
    return predicted_yield[0]

# Provide detailed suggestions based on user input
def provide_suggestions(predicted_yield, rain_fall, fertilizer, temperature, nitrogen, phosphorus, potassium):
    suggestions = []
    
    # Yield-related suggestions
    if predicted_yield > high_yield_threshold:
        suggestions.append(
            "The predicted yield is **above average**. This suggests that current farming practices, including "
            "fertilizer use and irrigation, are sufficient for a good yield. However, consider **reducing fertilizer** "
            "application slightly to optimize costs without sacrificing productivity."
        )
    else:
        if fertilizer < low_fertilizer_threshold:
            suggestions.append(
                "The predicted yield is **below average**, possibly due to insufficient fertilizer application. "
                "Consider **increasing fertilizer** usage, especially nitrogen-based fertilizers, which are crucial for plant growth. "
                "Ensure that any increase follows recommended guidelines to avoid over-fertilization and negative environmental impacts."
            )
        else:
            suggestions.append(
                "The predicted yield is **below average**. Fertilizer levels seem adequate, so it may be worthwhile to "
                "investigate other factors such as **soil health**, **pest management**, or **irrigation practices**. "
                "Testing the soil for deficiencies or adjusting planting techniques could help improve yield."
            )

    # Rainfall-related suggestions
    if rain_fall > high_rain_fall_threshold:
        suggestions.append(
            "The **rainfall** level is **high**, which could lead to problems like waterlogging or nutrient leaching. "
            "Ensure that your fields have good **drainage systems** to prevent soil erosion and root suffocation. "
            "You might also need to supplement nutrients that are washed away by heavy rain."
        )
    elif rain_fall < high_rain_fall_threshold:
        suggestions.append(
            "Rainfall levels seem moderate, but ensure your crops receive adequate water, especially during critical "
            "growth stages. Consider using **drip irrigation** or **mulching** to conserve moisture in the soil."
        )

    # Temperature-related suggestions
    if not (optimal_temperature_range[0] <= temperature <= optimal_temperature_range[1]):
        suggestions.append(
            f"The current temperature of **{temperature}°C** is **outside the optimal range** of {optimal_temperature_range[0]}-{optimal_temperature_range[1]}°C. "
            "If feasible, take steps to regulate the field temperature, such as **installing shade nets** or scheduling irrigation during cooler parts of the day. "
            "Extreme temperatures can stress crops, reducing their growth and productivity."
        )

    # Nitrogen-related suggestions
    if nitrogen < low_nitrogen_threshold:
        suggestions.append(
            "The **nitrogen** level is **low**, which is essential for leaf development and overall plant health. "
            "Consider applying nitrogen-rich fertilizers such as **urea** or **ammonium nitrate** to improve nitrogen levels. "
            "Be mindful of the application timing; for most crops, nitrogen is most beneficial during early vegetative growth stages."
        )
    elif nitrogen > high_nitrogen_threshold:
        suggestions.append(
            "The **nitrogen** level is **high**, which can lead to excessive vegetative growth at the expense of fruit or grain development. "
            "High nitrogen levels may also lead to increased pest and disease pressure. Consider **reducing nitrogen fertilization** "
            "to avoid potential issues, and focus on **balanced fertilization** strategies."
        )

    # Phosphorus-related suggestions
    if phosphorus < low_phosphorus_threshold:
        suggestions.append(
            "The **phosphorus** level is **low**, which is vital for root development and energy transfer within the plant. "
            "You may want to apply **phosphorus-based fertilizers** like **superphosphate** or **diammonium phosphate**. "
            "Phosphorus is particularly important early in the plant's lifecycle to promote strong root systems and overall plant vigor."
        )
    elif phosphorus > high_phosphorus_threshold:
        suggestions.append(
            "The **phosphorus** level is **high**, which can cause **nutrient imbalances** in the soil and reduce the plant's ability to absorb other essential nutrients like zinc or iron. "
            "Avoid further phosphorus application and consider **testing the soil** regularly to maintain a healthy balance."
        )

    # Potassium-related suggestions
    if potassium < low_potassium_threshold:
        suggestions.append(
            "The **potassium** level is **low**, which plays a crucial role in water regulation and crop quality. "
            "Consider applying **potassium-rich fertilizers** like **potash** or **sulfate of potash** to improve potassium levels. "
            "Adequate potassium can enhance drought resistance and increase crop quality, especially in fruits and vegetables."
        )
    elif potassium > high_potassium_threshold:
        suggestions.append(
            "The **potassium** level is **high**, which may affect the uptake of other essential nutrients like magnesium and calcium. "
            "Review your fertilization practices to maintain an optimal balance of nutrients for healthy plant growth."
        )

    return suggestions

# Streamlit UI
st.title("Crop Yield Prediction and Suggestion Tool")

# User input
rain_fall = st.sidebar.number_input("Rain Fall (mm)", min_value=0.0, step=0.1, value=100.0)
fertilizer = st.sidebar.number_input("Fertilizer", min_value=0.0, step=0.1, value=100.0)
temperature = st.sidebar.number_input("Temperature (°C)", min_value=0.0, step=0.1, value=25.0)
nitrogen = st.sidebar.number_input("Nitrogen (N)", min_value=0.0, step=0.1, value=50.0)
phosphorus = st.sidebar.number_input("Phosphorus (P)", min_value=0.0, step=0.1, value=20.0)
potassium = st.sidebar.number_input("Potassium (K)", min_value=0.0, step=0.1, value=20.0)

# Button to trigger prediction
if st.sidebar.button("Predict Crop Yield"):
    # Predict crop yield
    predicted_yield = predict_crop_yield(rain_fall, fertilizer, temperature, nitrogen, phosphorus, potassium)

    # Display result and suggestions
    st.write(f"Predicted Crop Yield: {predicted_yield:.2f} Q/acre")
    st.write("### Suggestions for improving crop yield:")

    suggestions = provide_suggestions(predicted_yield, rain_fall, fertilizer, temperature, nitrogen, phosphorus, potassium)
    for suggestion in suggestions:
        st.write(f"- {suggestion}")

    # Display progress bars for each parameter with ideal ranges
# Display progress bars for each parameter with ideal ranges
    def display_progress_bar(param_name, value, ideal_min, ideal_max, explanation):
        # Determine the bar color based on the value
        if value < ideal_min:
            bar_color = "red"  # Below the lower threshold
            displayed_value = ideal_min  # Set to the lower threshold
            fraction = f"{value}/{ideal_min}"  # Show the actual value in the numerator
        elif value > ideal_max:
            bar_color = "orange"  # Above the higher threshold
            displayed_value = ideal_max  # Set to the higher threshold
            fraction = f"{value}/{ideal_max}"  # Show the actual value in the numerator
        else:
            bar_color = "green"  # Within the optimal range
            displayed_value = value
            fraction = f"{value}/{ideal_max}"  # Normal case, display value over max
        
        # Display the fraction above the progress bar
        st.write(f"### {param_name}: {fraction}")

        # Calculate the progress for the progress bar
        progress = (value - ideal_min) / (ideal_max - ideal_min) if value <= ideal_max else 1.0

        # Display the progress bar with the appropriate color
        st.markdown(f"""
        <div style="width: 100%; background-color: #f3f3f3; border-radius: 10px;">
            <div style="width: {progress * 100}%; background-color: {bar_color}; height: 20px; border-radius: 10px;">
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Explanation of effects on crop yield
        st.write(f"**Effect on Crop Yield:** {explanation}")


# Display progress bars for Nitrogen, Phosphorus, and Potassium
    display_progress_bar("Phosphorus", phosphorus, low_phosphorus_threshold, high_phosphorus_threshold,
                        "Low phosphorus can hinder root development and reduce plant vigor, while high phosphorus may cause nutrient imbalances.")
    display_progress_bar("Nitrogen", nitrogen, low_nitrogen_threshold, high_nitrogen_threshold,
                        "Low nitrogen can lead to poor leaf development, while excessive nitrogen promotes excessive vegetative growth at the cost of fruit development.")
    display_progress_bar("Potassium", potassium, low_potassium_threshold, high_potassium_threshold,
                        "Low potassium affects water regulation and crop quality, while high potassium can hinder uptake of other essential nutrients like magnesium and calcium.")