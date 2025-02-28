import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import BytesIO
import requests
from PIL import Image

# ---------------------------
# (Unused) Utility Function: REFPROP
# ---------------------------
# (This function is no longer used since fluid properties will be manually provided.)
@st.cache_data
def cool(ref1, ref2, mf1, mf2, out, in1, valin1, in2, valin2):
    """Calculate thermodynamic properties using REFPROP (not used in this version)."""
    if ref2 == '' or pd.isna(ref2):
        fluid = 'REFPROP::' + ref1
    else:
        fluid = f'REFPROP::{ref1}&{ref2}'
    # Note: This is not used since we are not computing properties.
    return None

# ---------------------------
# App Title and Overview Image
# ---------------------------
st.title("🌡️ Heat Transfer Coefficient Predictor")
st.subheader("Model: **XGBoost with Optuna & PCA**")

google_drive_url = "https://drive.google.com/uc?export=download&id=1itd1HnJBWUEXGUq0B8sFJhmXjWEjtQ1t"
try:
    response = requests.get(google_drive_url)
    image = Image.open(BytesIO(response.content))
    st.image(image, caption='Heat Transfer Coefficient Model Overview', use_container_width=True)
except Exception as e:
    st.write(f"Error loading image: {e}")

# ---------------------------
# Model Loading Functions
# ---------------------------
@st.cache_resource
def load_xgb_model():
    xgb_url = "https://drive.google.com/uc?export=download&id=1VWVEgUC0HAKLuytfY_hxePfQ3Qk9BAT2"
    response = requests.get(xgb_url)
    response.raise_for_status()
    return joblib.load(BytesIO(response.content))

@st.cache_resource
def load_pca_model():
    pca_url = "https://drive.google.com/uc?export=download&id=1HHOaQgxUDbA6iPEAkQHh1gJvihz-MShn"
    response = requests.get(pca_url)
    response.raise_for_status()
    return joblib.load(BytesIO(response.content))

# ---------------------------
# Mode Selection
# ---------------------------
mode = st.radio("Select Mode", ("Single Data Point", "Batch Excel Upload"))

# ---------------------------
# Single Data Point Mode (Manual Fluid Property Input)
# ---------------------------
if mode == "Single Data Point":
    st.header("Single Data Point Prediction")
    
    st.subheader("1. Input Fluid Details")
    fluid1 = st.text_input("Enter primary fluid name (e.g., Water, R134a, etc.):", "Water")
    fluid2 = st.text_input("Enter secondary fluid name (or leave blank if none):", "")
    mf1 = st.number_input("Enter mass fraction of fluid 1 (0 to 1):", min_value=0.0, max_value=1.0, value=1.0, step=0.01)
    if fluid2:
        mf2 = 1.0 - mf1
    else:
        mf2 = 0

    st.subheader("2. Input Fluid Properties")
    Tsat = st.number_input("Saturation Temperature (Tsat) [K]:", value=313.0, format="%.2f")
    rho_l = st.number_input("Liquid Density (rho_l) [kg/m³]:", value=1000.0, format="%.2f")
    rho_v = st.number_input("Vapor Density (rho_v) [kg/m³]:", value=10.0, format="%.2f")
    mu_l = st.number_input("Liquid Viscosity (mu_l) [Pa.s]:", value=0.001, format="%.4f")
    mu_v = st.number_input("Vapor Viscosity (mu_v) [Pa.s]:", value=0.00001, format="%.6f")
    k_l = st.number_input("Liquid Thermal Conductivity (k_l) [W/mK]:", value=0.6, format="%.2f")
    k_v = st.number_input("Vapor Thermal Conductivity (k_v) [W/mK]:", value=0.02, format="%.2f")
    surface_tension = st.number_input("Surface Tension (N/m):", value=0.072, format="%.3f")
    Cp_l = st.number_input("Liquid Specific Heat (Cp_l) [J/kgK]:", value=4180.0, format="%.2f")
    Cp_v = st.number_input("Vapor Specific Heat (Cp_v) [J/kgK]:", value=2000.0, format="%.2f")
    Psat = st.number_input("Saturation Pressure (Psat) [Pa]:", value=101325.0, format="%.2f")
    
    st.subheader("3. Additional Inputs")
    x = st.number_input("Enter quality (x) (0 for liquid, 1 for vapor):", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    D = st.number_input("Enter diameter (m):", value=0.025, format="%.4f")
    G = st.number_input("Enter mass flux (G) in kg/m²s:", value=200.0, format="%.2f")
    
    if st.button("Calculate Heat Transfer Coefficient (h)"):
        # Create DataFrame for the 14 features (order must match your training data)
        input_dict = {
            'G (kg/m2s)': [G],
            'x': [x],
            'Tsat (K)': [Tsat],
            'rho_l': [rho_l],
            'rho_v': [rho_v],
            'mu_l': [mu_l],
            'mu_v': [mu_v],
            'k_v': [k_v],
            'k_l': [k_l],
            'surface_tension': [surface_tension],
            'Cp_v': [Cp_v],
            'Cp_l': [Cp_l],
            'Psat (Pa)': [Psat],
            'D (m)': [D]
        }
        input_data = pd.DataFrame(input_dict)
        
        # Log transformation (adding a small epsilon to avoid log(0))
        epsilon = 1e-10
        log_transformed_data = np.log(input_data + epsilon)
        
        # Load models from Google Drive
        pca = load_pca_model()
        xgb_model = load_xgb_model()
        
        # Apply PCA transformation and predict
        X_pca = pca.transform(log_transformed_data)
        predicted_log_h = xgb_model.predict(X_pca)
        predicted_h = np.exp(predicted_log_h)
        
        # Display results
        st.write("### Input Fluid Properties")
        st.dataframe(input_data)
        st.write(f"### The predicted heat transfer coefficient is: **{predicted_h[0]:.4f} W/m²K**")

# ---------------------------
# Batch Excel Upload Mode
# ---------------------------
elif mode == "Batch Excel Upload":
    st.header("Batch Processing from Excel File")
    st.info("Ensure your file columns are arranged (columnwise) as: Refrigerant 1, Refrigerant 2, Mass Fraction of Refrigerant 1, Saturation Temperature (K), Saturation Pressure (Pa), Quality (x), Diameter (m), Mass Flux (kg/m^2.s), Liquid Density, Vapor Density, Liquid Viscosity, Vapor Viscosity, Liquid Thermal Conductivity, Vapor Thermal Conductivity, Surface Tension, Liquid Specific Heat, Vapor Specific Heat")
    
    uploaded_file = st.file_uploader("Upload Excel or CSV file", type=["xlsx", "xls", "csv"])
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.write("### Uploaded Data:")
            st.dataframe(df)
            
            if st.button("Process Batch"):
                # Load pre-trained models (using local files in this mode)
                pca = joblib.load('pca_updated_model.pkl')
                xgb_model = joblib.load('updated_xgboost_model.pkl')
                
                predicted_htc_list = []  # To store predictions for each row
                
                for index, row in df.iterrows():
                    # Create DataFrame of features assuming the file already contains the 14 columns in the correct order:
                    features = pd.DataFrame({
                        'G (kg/m2s)': [row['G (kg/m2s)']],
                        'x': [row['x']],
                        'Tsat (K)': [row['Tsat (K)']],
                        'rho_l': [row['rho_l']],
                        'rho_v': [row['rho_v']],
                        'mu_l': [row['mu_l']],
                        'mu_v': [row['mu_v']],
                        'k_v': [row['k_v']],
                        'k_l': [row['k_l']],
                        'surface_tension': [row['surface_tension']],
                        'Cp_v': [row['Cp_v']],
                        'Cp_l': [row['Cp_l']],
                        'Psat (Pa)': [row['Psat (Pa)']],
                        'D (m)': [row['D (m)']]
                    })
                    
                    epsilon = 1e-10
                    log_features = np.log(features + epsilon)
                    
                    # PCA transformation and prediction
                    X_pca = pca.transform(log_features)
                    predicted_log_h = xgb_model.predict(X_pca)
                    predicted_h = np.exp(predicted_log_h)[0]
                    
                    predicted_htc_list.append(predicted_h)
                
                # Append predictions to the DataFrame and display
                df['Predicted HTC (W/m²K)'] = predicted_htc_list
                st.write("### Processed Data:")
                st.dataframe(df)
                
                # Create an Excel file in memory for download
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False, sheet_name='Results')
                    writer.save()
                processed_data = output.getvalue()
                
                st.download_button(
                    label="Download Processed Excel File",
                    data=processed_data,
                    file_name='processed_results.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )
        except Exception as e:
            st.error(f"Error processing file: {e}")
