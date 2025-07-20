import streamlit as st
import pandas as pd
import pickle

# === Load model and encoders ===
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)  # Dict: {column_name: LabelEncoder}

# === Helper function to preprocess user input ===
def preprocess_input(input_data: dict, model, default_value=0):
    input_df = pd.DataFrame([input_data])
    expected_features = model.feature_names_in_

    # Fill missing columns
    for col in expected_features:
        if col not in input_df.columns:
            input_df[col] = default_value

    # Reorder
    input_df = input_df[expected_features]
    return input_df

# === Streamlit UI ===
st.title("✈️ Flight Satisfaction Prediction")

# === Collect user input ===
gender = st.selectbox("Gender", ["Male", "Female"])
customer_type = st.selectbox("Customer Type", ["Loyal Customer", "Disloyal Customer"])
type_of_travel = st.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
class_type = st.selectbox("Class", ["Eco", "Eco Plus", "Business"])
age = st.slider("Age", 10, 80, 30)
flight_distance = st.number_input("Flight Distance", min_value=100, max_value=5000, value=1000)
inflight_wifi = st.slider("Inflight Wifi Service (0–5)", 0, 5, 3)
departure_delay = st.number_input("Departure Delay in Minutes", min_value=0, value=0)
arrival_delay = st.number_input("Arrival Delay in Minutes", min_value=0, value=0)

# Additional service ratings (you can extend these)
cleanliness = st.slider("Cleanliness", 0, 5, 3)
baggage_handling = st.slider("Baggage Handling", 0, 5, 3)
checkin_service = st.slider("Checkin Service", 0, 5, 3)

# === Predict button ===
if st.button("Predict Satisfaction"):
    # Step 1: Build input dict
    input_data = {
        "Gender": gender,
        "Customer Type": customer_type,
        "Type of Travel": type_of_travel,
        "Class": class_type,
        "Age": age,
        "Flight Distance": flight_distance,
        "Inflight Wifi Service": inflight_wifi,
        "Departure Delay in Minutes": departure_delay,
        "Arrival Delay in Minutes": arrival_delay,
        "Cleanliness": cleanliness,
        "Baggage handling": baggage_handling,
        "Checkin service": checkin_service
    }

    # Step 2: Apply Label Encoding if needed
    for col, encoder in label_encoders.items():
        if col in input_data:
            input_data[col] = encoder.transform([input_data[col]])[0]

    # Step 3: Align input features
    input_df = preprocess_input(input_data, model)

    # Step 4: Predict
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Satisfaction: {'Satisfied' if prediction == 1 else 'Neutral or Dissatisfied'}")
