import pickle
import joblib
import pandas as pd
import streamlit as st
import numpy as np

# ------------------ Page config ------------------
st.set_page_config(
    page_title="Chennai House Price Prediction",
    layout="centered"
)

st.title("🏠 Chennai House Price Prediction")

# ------------------ Load artifacts ------------------
@st.cache_resource
def load_encoders():
    with open("label_encoders.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

label_encoders = load_encoders()
model = load_model()

# ------------------ Input section ------------------
st.subheader("Enter Property Details")

area = st.selectbox(
    "Area",
    label_encoders["AREA"].classes_
)

park_facil = st.selectbox(
    "Park Facility Available?",
    label_encoders["PARK_FACIL"].classes_
)

build_type = st.selectbox(
    "Build Type",
    label_encoders["BUILDTYPE"].classes_
)

year_build = st.number_input(
    "Year Built",
    min_value=1900,
    max_value=2100,
    step=1
)

year_sale = st.number_input(
    "Year of Sale",
    min_value=1900,
    max_value=2100,
    step=1
)

int_sqft = st.number_input(
    "Interior SqFt",
    min_value=0,
    step=10
)

n_room = st.number_input(
    "Number of Rooms",
    min_value=0,
    step=1
)

reg_fee = st.number_input(
    "Registration Fee",
    min_value=0,
    step=1000
)

comm = st.number_input(
    "Commission Fee",
    min_value=0,
    step=1000
)

# ------------------ Prediction ------------------
if st.button("Predict Price"):
    # Create DataFrame
    new_df = pd.DataFrame({
        "AREA": [area],
        "PARK_FACIL": [park_facil],
        "BUILDTYPE": [build_type],
        "year_build": [year_build],
        "year_sale": [year_sale],
        "INT_SQFT": [int_sqft],
        "N_ROOM": [n_room],
        "REG_FEE": [reg_fee],
        "COMMIS": [comm]
    })

    # Encode categorical columns
    for col in ['PARK_FACIL', 'BUILDTYPE', 'AREA']:
        new_df[col] = label_encoders[col].transform(new_df[col])

    # Predict
    prediction = model.predict(new_df)
    price = np.exp(int(prediction[0]))
    if price >= 1_00_00_000:
        st.success(f"💰 Estimated Price: ₹ {price / 1_00_00_000:.2f} Crore")
    else:
        st.success(f"💰 Estimated Price: ₹ {price / 1_00_000:.2f} Lakh")

