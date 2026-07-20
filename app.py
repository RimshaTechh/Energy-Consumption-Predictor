import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Appliances Energy Predictor", layout="centered")
st.title("🏠 Appliances Energy Consumption Predictor")
st.write(
    "Enter the sensor readings below and this app will predict the "
    "energy consumption of appliances (in Wh) using a trained Random Forest model."
)

# -----------------------------
# Load model and feature list
# -----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    features = joblib.load("features.pkl")
    return model, features

try:
    model, feature_names = load_model()
except FileNotFoundError:
    st.error(
        "Could not find 'model.pkl' and/or 'features.pkl'. "
        "Make sure both files are in the same folder as this app."
    )
    st.stop()

# -----------------------------
# Sidebar inputs
# -----------------------------
st.sidebar.header("Input Features")

st.sidebar.subheader("Time")
hour = st.sidebar.slider("Hour of day", 0, 23, 12)
day_of_week = st.sidebar.selectbox(
    "Day of week", options=list(range(7)),
    format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday",
                            "Friday", "Saturday", "Sunday"][x]
)
is_weekend = 1 if day_of_week >= 5 else 0
month = st.sidebar.slider("Month", 1, 12, 6)

st.sidebar.subheader("Lights")
lights = st.sidebar.number_input("Lights energy use (Wh)", min_value=0, max_value=100, value=0)

st.sidebar.subheader("Indoor Temperature (°C)")
T1 = st.sidebar.slider("T1 (Kitchen)", 10.0, 30.0, 21.0)
T2 = st.sidebar.slider("T2 (Living room)", 10.0, 30.0, 20.0)
T3 = st.sidebar.slider("T3 (Laundry room)", 10.0, 30.0, 22.0)
T4 = st.sidebar.slider("T4 (Office room)", 10.0, 30.0, 21.0)
T5 = st.sidebar.slider("T5 (Bathroom)", 10.0, 30.0, 20.0)
T6 = st.sidebar.slider("T6 (Outside building, north)", -10.0, 35.0, 10.0)
T7 = st.sidebar.slider("T7 (Ironing room)", 10.0, 30.0, 20.0)
T8 = st.sidebar.slider("T8 (Teenager room)", 10.0, 30.0, 21.0)
T9 = st.sidebar.slider("T9 (Parents room)", 10.0, 30.0, 19.0)

st.sidebar.subheader("Indoor Humidity (%)")
RH_1 = st.sidebar.slider("RH_1 (Kitchen)", 20.0, 65.0, 40.0)
RH_2 = st.sidebar.slider("RH_2 (Living room)", 15.0, 60.0, 40.0)
RH_3 = st.sidebar.slider("RH_3 (Laundry room)", 25.0, 55.0, 40.0)
RH_4 = st.sidebar.slider("RH_4 (Office room)", 20.0, 55.0, 39.0)
RH_5 = st.sidebar.slider("RH_5 (Bathroom)", 25.0, 100.0, 45.0)
RH_6 = st.sidebar.slider("RH_6 (Outside building, north)", 0.0, 100.0, 55.0)
RH_7 = st.sidebar.slider("RH_7 (Ironing room)", 20.0, 55.0, 35.0)
RH_8 = st.sidebar.slider("RH_8 (Teenager room)", 20.0, 60.0, 42.0)
RH_9 = st.sidebar.slider("RH_9 (Parents room)", 25.0, 55.0, 41.0)

st.sidebar.subheader("Outdoor Weather")
T_out = st.sidebar.slider("Outdoor temperature (°C)", -10.0, 30.0, 7.0)
Press_mm_hg = st.sidebar.slider("Pressure (mm Hg)", 725.0, 775.0, 755.0)
RH_out = st.sidebar.slider("Outdoor humidity (%)", 20.0, 100.0, 80.0)
Windspeed = st.sidebar.slider("Windspeed (m/s)", 0.0, 15.0, 4.0)
Visibility = st.sidebar.slider("Visibility (km)", 0.0, 70.0, 38.0)
Tdewpoint = st.sidebar.slider("Dew point temperature (°C)", -10.0, 16.0, 4.0)

# -----------------------------
# Build input dataframe (must match training column order)
# -----------------------------
input_dict = {
    "lights": lights,
    "T1": T1, "RH_1": RH_1,
    "T2": T2, "RH_2": RH_2,
    "T3": T3, "RH_3": RH_3,
    "T4": T4, "RH_4": RH_4,
    "T5": T5, "RH_5": RH_5,
    "T6": T6, "RH_6": RH_6,
    "T7": T7, "RH_7": RH_7,
    "T8": T8, "RH_8": RH_8,
    "T9": T9, "RH_9": RH_9,
    "T_out": T_out,
    "Press_mm_hg": Press_mm_hg,
    "RH_out": RH_out,
    "Windspeed": Windspeed,
    "Visibility": Visibility,
    "Tdewpoint": Tdewpoint,
    "hour": hour,
    "day_of_week": day_of_week,
    "is_weekend": is_weekend,
    "month": month,
}

input_df = pd.DataFrame([input_dict])

# Reorder columns to exactly match what the model was trained on
try:
    input_df = input_df[feature_names]
except KeyError as e:
    st.error(f"Feature mismatch between app inputs and saved model features: {e}")
    st.stop()

# -----------------------------
# Prediction
# -----------------------------
st.subheader("Prediction")

if st.button("Predict Energy Consumption"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Appliances Energy Consumption: **{prediction:.2f} Wh**")

    with st.expander("See the exact inputs sent to the model"):
        st.dataframe(input_df.T.rename(columns={0: "Value"}))
else:
    st.info("Set the sensor values in the sidebar, then click **Predict Energy Consumption**.")

st.caption("Model: Random Forest Regressor trained on the UCI Appliances Energy Prediction dataset.")
