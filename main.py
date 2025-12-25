import pickle
import joblib
import pandas as pd
import numpy as np

# Load label encoders (DICT of encoders)
with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

# Load model
model = joblib.load("model.pkl")

# User inputs
area = input("Enter the area: ")
park_facil = input("Enter the Park Facility availability: ")
build_type = input("Enter the build type: ")
year_build = int(input("Enter year build: "))
year_sale = int(input("Enter year sale: "))
int_sqft = int(input("Enter the SqFt: "))
n_room = int(input("Enter no of rooms: "))
reg_fee = int(input("Enter Reg Fee: "))
comm = int(input("Enter Commission Fee: "))

# Create DataFrame (VALUES MUST BE LISTS)
new_df = pd.DataFrame({
    "AREA": [area],
    "PARK_FACIL": [park_facil],
    "BUILDTYPE": [build_type],
    'year_build':[year_build],
    'year_sale':[year_sale],
    "INT_SQFT": [int_sqft],
    "N_ROOM": [n_room],
    "REG_FEE": [reg_fee],
    "COMMIS": [comm]
})

# Categorical columns
categorical_columns = ['PARK_FACIL', 'BUILDTYPE', 'AREA']

# Encode categorical columns
for col in categorical_columns:
    new_df[col] = label_encoders[col].transform(new_df[col])

# Predict
prediction = model.predict(new_df)
price = np.exp(int(prediction[0]))
if price >= 1_00_00_000:
    print(f"💰 Estimated Price: ₹ {price / 1_00_00_000:.2f} Crore")
else:
    print(f"💰 Estimated Price: ₹ {price / 1_00_000:.2f} Lakh")
#Karapakkam	yes	Commercial	1967	2011	1004	3	380000	144400.0	7600000