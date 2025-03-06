import streamlit as st
import pandas as pd
import numpy as np
import CoolProp.CoolProp as CP
import joblib
from io import BytesIO
import requests
from PIL import Image
import matplotlib.pyplot as plt

import streamlit as st

# Define the CSS for dark and light themes
dark = """
    <style>
        .stApp {
            background-color: black;
            color: purple;
        }
    </style>
"""

light = """
    <style>
        .stApp {
            background-color: white;
            color: black;
        }
    </style>
"""

# Display the initial theme based on the session state
if "theme" not in st.session_state:
    st.session_state.theme = "light"  # Default theme is light

# Create a toggle button
toggle = st.button("Toggle theme")

# Change the theme based on the button state
if toggle:
    if st.session_state.theme == "light":
        st.session_state.theme = "dark"
    else:
        st.session_state.theme = "light"

# Apply the theme to the app
if st.session_state.theme == "dark":
    st.markdown(dark, unsafe_allow_html=True)
else:
    st.markdown(light, unsafe_allow_html=True)



# ---------------------------
# Utility Function: CoolProp Calculation using HEOS
# ---------------------------
@st.cache_data
def cool(ref1, ref2, mf1, mf2, out, in1, valin1, in2, valin2):
    """Calculate thermodynamic properties using CoolProp's HEOS backend."""
    if ref2 == '' or pd.isna(ref2):
        fluid = 'HEOS::' + ref1
    else:
        state = CP.AbstractState('HEOS', f'{ref1}&{ref2}')
        state.set_mass_fractions([mf1, mf2])
        fluid = f'HEOS::{ref1}&{ref2}'
    return CP.PropsSI(out, in1, valin1, in2, valin2, fluid)

# ---------------------------
# App Title and Overview Image
# ---------------------------
st.title("Condensation Heat Transfer Coefficient Predictor")
st.subheader("Model: **XGBoost with Optuna & PCA**")

google_drive_url = "https://drive.google.com/uc?export=download&id=1itd1HnJBWUEXGUq0B8sFJhmXjWEjtQ1t"
try:
    response = requests.get(google_drive_url)
    image = Image.open(BytesIO(response.content))
    st.image(image, caption='Heat Transfer Coefficient Model Overview', use_container_width=True)
except Exception as e:
    st.write(f"Error loading image: {e}")

with st.expander("🔍 Show Model Validity Information"):
    st.markdown("#### Model Training Data Ranges")
    st.markdown("""
    The model was trained with the following ranges:
    
    - **Mass Flux (G):** 24 to 1100 kg/m²s  
    - **Quality (x):** 0.01 to 0.99  
    - **Saturation Temperature (Tsat):** 242 to 356 K    
    - **Inner tube diameter (d):** 0.49 to 20 mm  
    - **Refrigerants:** 'R134A', 'R290', 'R22', 'R410A', 'R601', 'R1234ZE(E)', 
        'R513A', 'R32', 'R744', 'R1234YF', 'R404A', 'R12',
        'R1270', 'R450A', 'R245FA', 'R170', 'R407C',
        'R290/R170 (67.0/33.0%)', 'R290/R170 (33.0/67.0%)',
        'R455A', 'R245FA/R601 (45.0/55.0%)', 'R717', 'R152A', 'R600A',
        'R32/R125 (60.0/40.0%)', 'R32/R1234ZE(E) (25.0/75.0%)', 'R123', 'R125', 'R-E170',
        'R32/R1234ZE(E) (45.0/55.0%)', 'R-E170/R744 (61.0/39.0%)', 'R1234YF/R32 (77.0/23.0%)',
        'R1234YF/R32 (48.0/52.0%)', 'R-E170/R744 (79.0/21.0%)', 'R236EA'
    """)

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
# Creative Mode Selection
# ---------------------------
if 'mode' not in st.session_state:
    st.session_state.mode = None

st.write("### Select Your Mode")
col1, col2 = st.columns(2)
with col1:
    if st.button("Single Data Point 📝"):
        st.session_state.mode = "Single Data Point"
with col2:
    if st.button("Multiple Data 📊"):
        st.session_state.mode = "Multiple Data"

mode = st.session_state.mode

if mode is None:
    st.info("Please select a mode above to continue.")

# ---------------------------
# Single Data Point Mode
# ---------------------------
elif mode == "Single Data Point":
    st.header("Single Data Point Prediction")
    
    # Section 1: Basic Fluid Details
    st.subheader("1. Input Fluid Details")
    fluid1 = st.text_input("Enter primary fluid name (e.g., R1234ZE(E), R134a, etc.):", "R134a", key="fluid1")
    fluid2 = st.text_input("Enter secondary fluid name (or leave blank if none):", "", key="fluid2")
    mf1 = st.number_input("Enter mass fraction of fluid 1 (0 to 1):", min_value=0.0, max_value=1.0, value=1.0, step=0.01, key="mf1")
    if fluid2:
        mf2 = 1.0 - mf1
    else:
        mf2 = 0

    # Section 2: Choose Method for Fluid Property Input
    st.subheader("2. Select Method for Fluid Properties")
    prop_method = st.radio("Choose how to provide fluid properties:",
                           ["Calculate using CoolProp", "Input manually"], key="prop_method")
    
    # Flag to indicate if CoolProp calculations succeeded.
    prop_success = False

    if prop_method == "Calculate using CoolProp":
        # Ask: Temperature or Pressure?
        temp_or_press = st.radio("Would you like to enter Temperature (T) or Pressure (P)?",
                                 ["T", "P"], key="temp_or_press")
        # Ask for quality (for property calculation)
        quality_prop = st.number_input("Enter quality (x) for property calculation (0 for liquid, 1 for vapor):",
                                       min_value=0.0, max_value=1.0, value=0.50, step=0.01, key="quality_prop")
        if temp_or_press == "T":
            T_input = st.number_input("Enter Temperature (K):", value=313.0, format="%.2f", key="T_input_calc")
        else:
            P_input = st.number_input("Enter Pressure (Pa):", value=101325.0, format="%.2f", key="P_input_calc")
        # Ask for Diameter and Mass Flux with unique keys
        D = st.number_input("Enter diameter (m):", value=0.0050, format="%.4f", key="D_calc")
        G = st.number_input("Enter mass flux (G) in kg/m²s:", value=200.00, format="%.2f", key="G_calc")
        
        try:
            if temp_or_press == "T":
                Psat = cool(fluid1, fluid2, mf1, mf2, 'P', 'T', T_input, 'Q', 0)
                rho_l = cool(fluid1, fluid2, mf1, mf2, 'D', 'T', T_input, 'Q', 0)
                rho_v = cool(fluid1, fluid2, mf1, mf2, 'D', 'T', T_input, 'Q', 1)
                mu_l = cool(fluid1, fluid2, mf1, mf2, 'V', 'T', T_input, 'Q', 0)
                mu_v = cool(fluid1, fluid2, mf1, mf2, 'V', 'T', T_input, 'Q', 1)
                k_l = cool(fluid1, fluid2, mf1, mf2, 'L', 'T', T_input, 'Q', 0)
                k_v = cool(fluid1, fluid2, mf1, mf2, 'L', 'T', T_input, 'Q', 1)
                surface_tension = cool(fluid1, fluid2, mf1, mf2, 'I', 'T', T_input, 'Q', quality_prop)
                Cp_l = cool(fluid1, fluid2, mf1, mf2, 'C', 'T', T_input, 'Q', 0)
                Cp_v = cool(fluid1, fluid2, mf1, mf2, 'C', 'T', T_input, 'Q', 1)
            else:
                T_input = cool(fluid1, fluid2, mf1, mf2, 'T', 'P', P_input, 'Q', 0)
                Psat = P_input
                rho_l = cool(fluid1, fluid2, mf1, mf2, 'D', 'T', T_input, 'Q', 0)
                rho_v = cool(fluid1, fluid2, mf1, mf2, 'D', 'T', T_input, 'Q', 1)
                mu_l = cool(fluid1, fluid2, mf1, mf2, 'V', 'T', T_input, 'Q', 0)
                mu_v = cool(fluid1, fluid2, mf1, mf2, 'V', 'T', T_input, 'Q', 1)
                k_l = cool(fluid1, fluid2, mf1, mf2, 'L', 'T', T_input, 'Q', 0)
                k_v = cool(fluid1, fluid2, mf1, mf2, 'L', 'T', T_input, 'Q', 1)
                surface_tension = cool(fluid1, fluid2, mf1, mf2, 'I', 'T', T_input, 'Q', quality_prop)
                Cp_l = cool(fluid1, fluid2, mf1, mf2, 'C', 'T', T_input, 'Q', 0)
                Cp_v = cool(fluid1, fluid2, mf1, mf2, 'C', 'T', T_input, 'Q', 1)
            prop_success = True
        except Exception as e:
            st.error("CoolProp failed to calculate properties. Please input properties manually.")
    
    if prop_method == "Input manually" or not prop_success:
        st.info("Please manually input the fluid properties:")
        T_input = st.number_input("Enter Saturation Temperature (Tsat) [K]:", value=313.0, format="%.2f", key="T_input_manual")
        rho_l = st.number_input("Liquid Density (rho_l) [kg/m³]:", value=1000.0, format="%.2f", key="rho_l_manual")
        rho_v = st.number_input("Vapor Density (rho_v) [kg/m³]:", value=10.0, format="%.2f", key="rho_v_manual")
        mu_l = st.number_input("Liquid Viscosity (mu_l) [Pa.s]:", value=0.001, format="%.4f", key="mu_l_manual")
        mu_v = st.number_input("Vapor Viscosity (mu_v) [Pa.s]:", value=0.00001, format="%.6f", key="mu_v_manual")
        k_l = st.number_input("Liquid Thermal Conductivity (k_l) [W/mK]:", value=0.6, format="%.2f", key="k_l_manual")
        k_v = st.number_input("Vapor Thermal Conductivity (k_v) [W/mK]:", value=0.02, format="%.2f", key="k_v_manual")
        surface_tension = st.number_input("Surface Tension (N/m):", value=0.072, format="%.3f", key="surf_manual")
        Cp_l = st.number_input("Liquid Specific Heat (Cp_l) [J/kgK]:", value=4180.0, format="%.2f", key="Cp_l_manual")
        Cp_v = st.number_input("Vapor Specific Heat (Cp_v) [J/kgK]:", value=2000.0, format="%.2f", key="Cp_v_manual")
        Psat = st.number_input("Saturation Pressure (Psat) [Pa]:", value=101325.0, format="%.2f", key="Psat_manual")
        D = st.number_input("Enter diameter (m):", value=0.0050, format="%.4f", key="D_manual")
        G = st.number_input("Enter mass flux (G) in kg/m²s:", value=200.00, format="%.2f", key="G_manual")
    
    # Section 3: Quality input (if not already captured)
    if prop_method == "Input manually" or not prop_success:
        x_val = st.number_input("Enter quality (x) (0 for liquid, 1 for vapor):", min_value=0.0, max_value=1.0, value=0.5, step=0.01, key="x_manual")
    else:
        x_val = quality_prop

    if st.button("Calculate Heat Transfer Coefficient (h)"):
        # Assemble feature DataFrame in the order:
        # G (kg/m2s), x, Tsat (K), rho_l, rho_v, mu_l, mu_v, k_v, k_l, surface_tension, Cp_v, Cp_l, Psat (Pa), D (m)
        diameter = D
        mass_flux = G
        feature_dict = {
            'G (kg/m2s)': [mass_flux],
            'x': [x_val],
            'Tsat (K)': [T_input],
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
            'D (m)': [diameter]
        }
        input_data = pd.DataFrame(feature_dict)
        epsilon = 1e-10
        log_transformed_data = np.log(input_data + epsilon)
        pca = load_pca_model()
        xgb_model = load_xgb_model()
        X_pca = pca.transform(log_transformed_data)
        predicted_log_h = xgb_model.predict(X_pca)
        predicted_h = np.exp(predicted_log_h)
        st.write("### Fluid Properties Used")
        st.dataframe(input_data)
        st.write(f"### <span style='color:blue;'>The predicted heat transfer coefficient is: **{predicted_h[0]:.4f} W/m²K**</span>", unsafe_allow_html=True)
        st.info("The Mean Absolute Percentage Error of the model is 9.22 %")

# ---------------------------
# Multiple Data Mode
# ---------------------------
# ---------------------------



elif mode == "Multiple Data":
    st.header("Multiple Data Processing")
    st.info("Ensure your file includes all required fluid properties in the following order:\n"
            "Mass Flux (kg/m^2.s), Quality (x), Saturation Temperature (K), Density of liquid phase (kg/m^3), "
            "Density of vapor phase (kg/m^3), Dynamic viscosity of liquid phase (Ns/m^2), Dynamic viscosity of vapor phase (Ns/m^2), "
            "Thermal conductivity of vapor phase (W/m.K), Thermal conductivity of liquid phase (W/m.K), "
            "Surface Tension (N/m), Mass-specific constant pressure-specific heat of vapor phase (J/kg.K), "
            "Mass-specific constant pressure-specific heat of liquid phase (J/kg.K), Saturation pressure (Pa), Diameter (m)")
    
    uploaded_file = st.file_uploader("Upload Excel or CSV file", type=["xlsx", "xls", "csv"])
    if uploaded_file is not None:
        try:
            col_names = [
                'G (kg/m2s)', 'x', 'Tsat (K)', 'rho_l', 'rho_v', 
                'mu_l', 'mu_v', 'k_v', 'k_l', 'surface_tension', 
                'Cp_v', 'Cp_l', 'Psat (Pa)', 'D (m)'
            ]
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, header=None, skiprows=1, names=col_names)
            else:
                df = pd.read_excel(uploaded_file, engine='openpyxl', header=None, skiprows=1, names=col_names)
            
            st.write("### Uploaded Data:")
            st.dataframe(df)
            if st.button("Process Multiple Data"):
                pca = load_pca_model()
                xgb_model = load_xgb_model()
                predicted_htc_list = []
                for index, row in df.iterrows():
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
                    X_pca = pca.transform(log_features)
                    predicted_log_h = xgb_model.predict(X_pca)
                    predicted_h = np.exp(predicted_log_h)[0]
                    predicted_htc_list.append(predicted_h)
                df['Predicted HTC (W/m²K)'] = predicted_htc_list
                st.write("### Processed Data:")
                st.dataframe(df)
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False, sheet_name='Results')
                processed_data = output.getvalue()
                st.download_button(
                    label="Download Processed Excel File",
                    data=processed_data,
                    file_name='processed_results.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )
                st.info("The Mean Absolute Percentage Error of the model is 9.22 %")
                # Save processed DataFrame to session state for later use
                st.session_state["df_processed"] = df
                
        except Exception as e:
            st.error(f"Error processing file: {e}")
    
    # Graph Generation Section (accessible if processed DataFrame exists)
    if "df_processed" in st.session_state:
        st.write("### Generate Graph")
        processed_df = st.session_state["df_processed"]
        x_var = st.selectbox("Select variable for X-axis", options=processed_df.columns, key="x_axis")
        y_var = st.selectbox("Select variable for Y-axis", options=processed_df.columns, key="y_axis")
        if st.button("Generate Graph"):
            fig, ax = plt.subplots()
            ax.scatter(processed_df[x_var], processed_df[y_var])
            ax.set_xlabel(x_var)
            ax.set_ylabel(y_var)
            ax.set_title(f"{y_var} vs {x_var}")
            st.pyplot(fig)
            
            # Save the graph as PNG and provide a download button
            buf = BytesIO()
            fig.savefig(buf, format="png",  bbox_inches='tight', dpi=800)
            buf.seek(0)
            st.download_button(
                label="Download Graph as PNG",
                data=buf,
                file_name="graph.png",
                mime="image/png"
            )
