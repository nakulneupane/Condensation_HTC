import streamlit as st
import pandas as pd
import numpy as np
import CoolProp.CoolProp as CP
import joblib
from io import BytesIO
import requests
from PIL import Image

# ---------------------------
# Utility Function: REFPROP
# ---------------------------
@st.cache_data
def cool(ref1, ref2, mf1, mf2, out, in1, valin1, in2, valin2):
    """Calculate thermodynamic properties using REFPROP (cached to avoid recomputation)."""
    if ref2 == '' or pd.isna(ref2):
        fluid = 'REFPROP::' + ref1
    else:
        state = CP.AbstractState('REFPROP', f'{ref1}&{ref2}')
        state.set_mass_fractions([mf1, mf2])
        mole_fractions = state.get_mole_fractions()
        fluid = f'REFPROP::{ref1}[{mole_fractions[0]:.4f}]&{ref2}[{mole_fractions[1]:.4f}]'
    result = CP.PropsSI(out, in1, valin1, in2, valin2, fluid)
    return result

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
    # Construct a direct download link from Google Drive
    xgb_url = "https://drive.google.com/uc?export=download&id=1VWVEgUC0HAKLuytfY_hxePfQ3Qk9BAT2"
    response = requests.get(xgb_url)
    response.raise_for_status()  # Raise an error for bad responses
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
# Single Data Point Mode
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

    st.header("2. Input Temperature or Pressure")
    input_type = st.radio("Would you like to enter Temperature (T) or Pressure (P)?", ('T', 'P'))
    if input_type == 'T':
        T = st.number_input("Enter temperature (K):", value=313.0, format="%.2f")
        Psat = cool(fluid1, fluid2, mf1, mf2, 'P', 'T', T, 'Q', 0)
    else:
        P = st.number_input("Enter pressure (Pa):", value=101325.0, format="%.2f")
        T = cool(fluid1, fluid2, mf1, mf2, 'T', 'P', P, 'Q', 0)
        Psat = P  # If pressure is provided, it's assumed as Psat

    st.header("3. Additional Inputs")
    x = st.number_input("Enter quality (x) (0 for liquid, 1 for vapor):", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    D = st.number_input("Enter diameter (m):", value=0.025, format="%.4f")
    G = st.number_input("Enter mass flux (G) in kg/m²s:", value=200.0, format="%.2f")
    
    if st.button("Calculate Heat Transfer Coefficient (h)"):
        # Calculate thermodynamic properties using REFPROP
        rho_l = cool(fluid1, fluid2, mf1, mf2, 'D', 'T', T, 'Q', 0)
        rho_v = cool(fluid1, fluid2, mf1, mf2, 'D', 'T', T, 'Q', 1)
        mu_l = cool(fluid1, fluid2, mf1, mf2, 'V', 'T', T, 'Q', 0)
        mu_v = cool(fluid1, fluid2, mf1, mf2, 'V', 'T', T, 'Q', 1)
        k_l = cool(fluid1, fluid2, mf1, mf2, 'L', 'T', T, 'Q', 0)
        k_v = cool(fluid1, fluid2, mf1, mf2, 'L', 'T', T, 'Q', 1)
        surface_tension = cool(fluid1, fluid2, mf1, mf2, 'I', 'T', T, 'Q', x)
        Cp_l = cool(fluid1, fluid2, mf1, mf2, 'C', 'T', T, 'Q', 0)
        Cp_v = cool(fluid1, fluid2, mf1, mf2, 'C', 'T', T, 'Q', 1)

        # Create a DataFrame for the 14 features
        input_dict = {
            'G (kg/m2s)': [G],
            'x': [x],
            'Tsat (K)': [T],
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
        
        # Log transformation (with epsilon to avoid log(0))
        epsilon = 1e-10
        log_transformed_data = np.log(input_data + epsilon)
        
        # Load models from Google Drive
        pca = load_pca_model()
        xgb_model = load_xgb_model()
        
        # Apply PCA transformation and predict
        X_pca = pca.transform(log_transformed_data)
        predicted_log_h = xgb_model.predict(X_pca)
        predicted_h = np.exp(predicted_log_h)
        
        # Display the results
        st.write("### Calculated Thermodynamic Properties")
        st.dataframe(input_data)
        st.write(f"### The predicted heat transfer coefficient is: **{predicted_h[0]:.4f} W/m²K**")

# ---------------------------
# Batch Excel Upload Mode
# ---------------------------
elif mode == "Batch Excel Upload":
    st.header("Batch Processing from Excel File")
    st.info("Ensure your file columns are arranged (columnwise) as: Refrigerant 1, Refrigerant 2, Mass Fraction of Refrigerant 1, Saturation Temperature (K), Saturation Pressure (Pa), Quality (x), Diameter (m), Mass Flux (kg/m^2.s)")
    
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
                    # Extract fluid and mass fraction details
                    fluid1 = row.iloc[0]
                    fluid2 = row.iloc[1] if pd.notna(row.iloc[1]) else ""
                    mf1 = row.iloc[2]
                    mf2 = 1.0 - mf1 if fluid2 != "" else 0
                    
                    # Determine Temperature or Pressure input
                    if pd.notna(row.iloc[3]):
                        T = row.iloc[3]
                        Psat = cool(fluid1, fluid2, mf1, mf2, 'P', 'T', T, 'Q', 0)
                    else:
                        P = row.iloc[4]
                        T = cool(fluid1, fluid2, mf1, mf2, 'T', 'P', P, 'Q', 0)
                        Psat = P  # When pressure is provided, it is assumed as Psat
                    
                    x_val = row.iloc[5]  # Quality (x)
                    D = row.iloc[6]      # Diameter
                    G = row.iloc[7]      # Mass Flux
                    
                    # Calculate thermodynamic properties using REFPROP
                    rho_l = cool(fluid1, fluid2, mf1, mf2, 'D', 'T', T, 'Q', 0)
                    rho_v = cool(fluid1, fluid2, mf1, mf2, 'D', 'T', T, 'Q', 1)
                    mu_l = cool(fluid1, fluid2, mf1, mf2, 'V', 'T', T, 'Q', 0)
                    mu_v = cool(fluid1, fluid2, mf1, mf2, 'V', 'T', T, 'Q', 1)
                    k_l = cool(fluid1, fluid2, mf1, mf2, 'L', 'T', T, 'Q', 0)
                    k_v = cool(fluid1, fluid2, mf1, mf2, 'L', 'T', T, 'Q', 1)
                    surface_tension = cool(fluid1, fluid2, mf1, mf2, 'I', 'T', T, 'Q', x_val)
                    Cp_l = cool(fluid1, fluid2, mf1, mf2, 'C', 'T', T, 'Q', 0)
                    Cp_v = cool(fluid1, fluid2, mf1, mf2, 'C', 'T', T, 'Q', 1)
                    
                    # Create DataFrame of features
                    features = pd.DataFrame({
                        'G (kg/m2s)': [G],
                        'x': [x_val],
                        'Tsat (K)': [T],
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
