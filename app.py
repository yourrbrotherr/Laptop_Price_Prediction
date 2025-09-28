import streamlit as st
import pandas as pd
import numpy as np
import pickle



# Load model artifacts
try:
    with open("laptop_price_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("encoders.pkl", "rb") as f:
        encoders = pickle.load(f)

    with open("feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)
except FileNotFoundError:
    st.error("Model or encoder files not found. Please ensure 'laptop_price_model.pkl', 'encoders.pkl', and 'feature_columns.pkl' are in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()


st.set_page_config(page_title="Laptop Price Predictor", layout="centered")
st.title("💻 Laptop Price Prediction")

st.markdown("""
This app predicts the price of a laptop based on its specifications.
""")


def get_options(feature_key):
    """Returns the list of original labels (classes) for a given LabelEncoder."""
    
    return encoders[feature_key].classes_



with st.container(border=True):
    st.subheader("General Specifications")
    
   
    Company = st.selectbox("Company", options=get_options('Company'))
    TypeName = st.selectbox("Type Name", options=get_options('TypeName'))
    OS = st.selectbox("Operating System", options=get_options('OS'))

    # Numeric Inputs
    cols_general = st.columns(3)
    with cols_general[0]:
        Ram = st.number_input("RAM (GB)", min_value=2, max_value=64, value=8, step=1, help="Total system memory in Gigabytes.")
    with cols_general[1]:
        Weight = st.number_input("Weight (kg)", min_value=0.5, max_value=5.0, value=1.5, step=0.1, format="%.1f", help="Weight of the laptop in kilograms.")
    with cols_general[2]:
        Inches = st.number_input("Screen Size (inches)", min_value=10.0, max_value=20.0, value=15.6, step=0.1, format="%.1f", help="Diagonal screen size.")

with st.container(border=True):
    st.subheader("Screen Details")
    
    ScreenType = st.selectbox("Screen Type/Resolution", options=get_options('Screen'))
    
    cols_screen = st.columns(3)
    with cols_screen[0]:
        ScreenW = st.number_input("Width (pixels)", min_value=800, max_value=4000, value=1920, step=1, help="Screen resolution width.")
    with cols_screen[1]:
        ScreenH = st.number_input("Height (pixels)", min_value=600, max_value=4000, value=1080, step=1, help="Screen resolution height.")

    # Boolean/Binary Inputs
    cols_features = st.columns(3)
    with cols_features[0]:
        Touchscreen = st.radio("Touchscreen", ["No", "Yes"], horizontal=True)
    with cols_features[1]:
        RetinaDisplay = st.radio("Retina Display", ["No", "Yes"], horizontal=True)
    with cols_features[2]:
        IPSpanel = st.radio("IPS Panel", ["No", "Yes"], horizontal=True)


with st.container(border=True):
    st.subheader("Processor & Graphics")

    cols_cpu = st.columns(2)
    with cols_cpu[0]:
        CPU_company = st.selectbox("CPU Company", options=get_options('CPU_company'))
        CPU_model = st.selectbox("CPU Model", options=get_options('CPU_model'))
    with cols_cpu[1]:
        GPU_company = st.selectbox("GPU Company", options=get_options('GPU_company'))
        GPU_model = st.selectbox("GPU Model", options=get_options('GPU_model'))
    
    CPU_freq = st.number_input("CPU Frequency (GHz)", min_value=1.0, max_value=5.0, value=2.5, step=0.1, format="%.1f", help="Base clock speed of the CPU.")


with st.container(border=True):
    st.subheader("Storage")
    
    cols_storage = st.columns(2)
    with cols_storage[0]:
        PrimaryStorageType = st.selectbox("Primary Storage Type", options=get_options('PrimaryStorageType'))
        PrimaryStorage = st.number_input("Primary Storage Size (GB)", min_value=0, max_value=4096, value=512, step=1, help="Size of the main drive (e.g., SSD).")
    with cols_storage[1]:
        SecondaryStorageType = st.selectbox("Secondary Storage Type", options=get_options('SecondaryStorageType'))
        SecondaryStorage = st.number_input("Secondary Storage Size (GB)", min_value=0, max_value=4096, value=0, step=1, help="Size of the secondary drive (e.g., HDD). Enter 0 if none.")



if st.button("Predict Price 💰", type="primary", use_container_width=True):
    
    # 1. Convert human-readable labels back to integers using .transform()
    try:
        input_dict = {
            'Company': encoders['Company'].transform([Company])[0],
            'TypeName': encoders['TypeName'].transform([TypeName])[0],
            'OS': encoders['OS'].transform([OS])[0],
            'Screen': encoders['Screen'].transform([ScreenType])[0],
            'CPU_company': encoders['CPU_company'].transform([CPU_company])[0],
            'CPU_model': encoders['CPU_model'].transform([CPU_model])[0],
            'PrimaryStorageType': encoders['PrimaryStorageType'].transform([PrimaryStorageType])[0],
            'SecondaryStorageType': encoders['SecondaryStorageType'].transform([SecondaryStorageType])[0],
            'GPU_company': encoders['GPU_company'].transform([GPU_company])[0],
            'GPU_model': encoders['GPU_model'].transform([GPU_model])[0],
            
            # Binary/Numeric features are already handled correctly
            'Touchscreen': 1 if Touchscreen == "Yes" else 0,
            'RetinaDisplay': 1 if RetinaDisplay == "Yes" else 0,
            'IPSpanel': 1 if IPSpanel == "Yes" else 0,
            'Inches': Inches,
            'Weight': Weight,
            'Ram': Ram,
            'CPU_freq': CPU_freq,
            'ScreenW': ScreenW,
            'ScreenH': ScreenH,
            'PrimaryStorage': PrimaryStorage,
            'SecondaryStorage': SecondaryStorage
        }

        # 2. Create DataFrame for prediction
        input_df = pd.DataFrame([input_dict])

        # 3. Ensure all feature columns are present in the correct order
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)

        # 4. Predict
        prediction = model.predict(input_df)[0]
        
        # 5. Display Result
        st.balloons()
        st.success("---")
        st.markdown(f"""
            ## Estimated Price: <span style='color: green;'>€{prediction:,.2f}</span>
            """, unsafe_allow_html=True)
        st.success("---")

    except ValueError as e:
        st.error(f"One of the selected categorical labels is not recognized by the model encoder. Please check input files. Error: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred during prediction: {e}")

    # 6. Show predicted price
    st.write(f"💰 Predicted Laptop Price: €{prediction:,.2f}")