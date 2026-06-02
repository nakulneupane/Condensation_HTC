import streamlit as st
import pandas as pd
import numpy as np
import CoolProp.CoolProp as CP
import joblib
from io import BytesIO
import requests
from PIL import Image
import matplotlib.pyplot as plt

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

st.title("Condensation Heat Transfer Coefficient Predictor")
st.subheader("Model: **XGBoost with Scikit-Optimize & PCA**")

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

@st.cache_resource
def load_xgb_skopt_model():
    """Load the XGBoost model tuned with scikit-optimize (works on PCA-transformed features)."""
    xgb_skopt_url = "https://drive.google.com/uc?export=download&id=1Zhv11sFtx0-Ww4gx3Z4xQiUgZHFIbVDH"
    response = requests.get(xgb_skopt_url)
    response.raise_for_status()
    model = joblib.load(BytesIO(response.content))
    return model

@st.cache_resource
def load_pca_model():
    """Load the PCA transformer."""
    pca_url = "https://drive.google.com/uc?export=download&id=1HHOaQgxUDbA6iPEAkQHh1gJvihz-MShn"
    response = requests.get(pca_url)
    response.raise_for_status()
    pca = joblib.load(BytesIO(response.content))
    return pca

# Mode Selection
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

# Single Data Point Mode
elif mode == "Single Data Point":
    st.header("Single Data Point Prediction")
    
    # Fluid Details
    st.subheader("1. Input Fluid Details")
    fluid1 = st.text_input("Enter primary fluid name (e.g., R1234ZE(E), R134a, etc.):", "R134a", key="fluid1")
    fluid2 = st.text_input("Enter secondary fluid name (or leave blank if none):", "", key="fluid2")
    mf1 = st.number_input("Enter mass fraction of fluid 1 (0 to 1):", min_value=0.0, max_value=1.0, value=1.0, step=0.01, key="mf1")
    mf2 = 1.0 - mf1 if fluid2 else 0.0

    # Fluid Property Input
    st.subheader("2. Select Method for Fluid Properties")
    prop_method = st.radio("Choose how to provide fluid properties:",
                           ["Calculate using CoolProp", "Input manually"], key="prop_method")
    
    prop_success = False

    if prop_method == "Calculate using CoolProp":
        temp_or_press = st.radio("Would you like to enter Temperature (T) or Pressure (P)?",
                                 ["T", "P"], key="temp_or_press")
        quality_prop = st.number_input("Enter quality (x) for property calculation (0 for liquid, 1 for vapor):",
                                       min_value=0.0, max_value=1.0, value=0.50, step=0.01, key="quality_prop")
        x_val = quality_prop
        if temp_or_press == "T":
            T_input = st.number_input("Enter Temperature (K):", value=313.0, format="%.2f", key="T_input_calc")
        else:
            P_input = st.number_input("Enter Pressure (Pa):", value=101325.0, format="%.2f", key="P_input_calc")
        
        D = st.number_input("Enter diameter (m):", value=0.0050, format="%.4f", key="D_calc")
        G = st.number_input("Enter mass flux (G) in kg/m²s:", value=200.00, format="%.2f", key="G_calc")
        
        try:
            # Calculating all properties
            if temp_or_press == "T":
                Psat = cool(fluid1, fluid2, mf1, mf2, 'P', 'T', T_input, 'Q', 0)
            else:
                Psat = P_input

            T_bubble = cool(fluid1, fluid2, mf1, mf2, 'T', 'P', Psat, 'Q', 0)
            T_dew = cool(fluid1, fluid2, mf1, mf2, 'T', 'P', Psat, 'Q', 1)
            glide = T_dew - T_bubble
            T_ref = 0.5 * (T_bubble + T_dew)
            rho_l = cool(fluid1, fluid2, mf1, mf2, 'D', 'T', T_ref, 'Q', 0)
            rho_v = cool(fluid1, fluid2, mf1, mf2, 'D', 'T', T_ref, 'Q', 1)
            mu_l = cool(fluid1, fluid2, mf1, mf2, 'V', 'T', T_ref, 'Q', 0)
            mu_v = cool(fluid1, fluid2, mf1, mf2, 'V', 'T', T_ref, 'Q', 1)
            k_l = cool(fluid1, fluid2, mf1, mf2, 'L', 'T', T_ref, 'Q', 0)
            k_v = cool(fluid1, fluid2, mf1, mf2, 'L', 'T', T_ref, 'Q', 1)
            surface_tension = cool(fluid1, fluid2, mf1, mf2, 'I', 'T', T_ref, 'Q', quality_prop)
            Cp_l = cool(fluid1, fluid2, mf1, mf2, 'C', 'T', T_ref, 'Q', 0)
            Cp_v = cool(fluid1, fluid2, mf1, mf2, 'C', 'T', T_ref, 'Q', 1)
            h_l = cool(fluid1, fluid2, mf1, mf2, 'H', 'T', T_ref, 'Q', 0)
            h_v = cool(fluid1, fluid2, mf1, mf2, 'H', 'T', T_ref, 'Q', 1)
            h_lv = h_v - h_l
            x_i = quality_prop
            Z = (x_i * Cp_v * glide) / h_lv if h_lv != 0 else 0.0
            R_m = Z / h_v

            prop_success = True

            st.success("Thermodynamic properties successfully calculated using CoolProp.")
            st.write(f"**Glide temperature:** {glide:.4f} K")
            st.write(f"**Latent heat (h_lv):** {h_lv:.2f} J/kg")
            st.write(f"**Z factor:** {Z:.6f}")
            st.write(f"**Mass transfer resistance (R_m):** {R_m:.8f}")

        except Exception as e:
            st.error(f"CoolProp failed to calculate properties: {e}")
            prop_success = False

    # Manual input
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
        x_val = st.number_input("Enter quality (x)", min_value=0.0, max_value=1.0, value=0.5, step=0.01, key="x_manual")
        
        st.markdown("### Additional inputs for Z-factor calculation")
        
        T_bubble = st.number_input("Bubble Temperature (K)", value=313.0, format="%.2f", key="T_bubble_manual")
        T_dew = st.number_input("Dew Temperature (K)", value=313.0, format="%.2f", key="T_dew_manual")
        h_l = st.number_input("Liquid Enthalpy h_l (J/kg)", value=200000.0, format="%.2f", key="hl_manual")
        h_v = st.number_input("Vapor Enthalpy h_v (J/kg)", value=400000.0, format="%.2f", key="hv_manual")
        
        glide = T_dew - T_bubble
        h_lv = h_v - h_l
        
        Z = (x_val * Cp_v * glide) / h_lv if h_lv != 0 else 0.0
        
        st.success("Z-factor calculated successfully")
        st.write(f"**Temperature Glide:** {glide:.4f} K")
        st.write(f"**Latent Heat (h_lv):** {h_lv:.2f} J/kg")
        st.write(f"**Z Factor:** {Z:.6f}")

    if st.button("Calculate Heat Transfer Coefficient (h)"):
        # Define the exact feature names in the correct order from PCA
        feature_names = [
            'G (kg/m2s)', 'x', 'Tsat (K)', 'rho_l', 'rho_v', 'mu_l', 'mu_v', 
            'k_v', 'k_l', 'surface_tension', 'Cp_v', 'Cp_l', 'Z', 'Psat (Pa)', 'D (m)'
        ]
        
        # Create array with values in the exact same order
        feature_values = np.array([[
            G, x_val, T_input, rho_l, rho_v, mu_l, mu_v, k_v, k_l, 
            surface_tension, Cp_v, Cp_l, Z, Psat, D
        ]])
        
        # Apply log transform (add epsilon to avoid log(0))
        epsilon = 1e-10
        log_transformed_data = np.log(feature_values + epsilon)
        
        # Create DataFrame with exact column names in exact order
        feature_df = pd.DataFrame(log_transformed_data, columns=feature_names)
        
        try:
            # Load models
            pca = load_pca_model()
            xgb_model = load_xgb_skopt_model()
            
            # Apply PCA transformation
            X_pca = pca.transform(feature_df)
            
            # Make prediction
            predicted_log_h = xgb_model.predict(X_pca)
            predicted_h = np.exp(predicted_log_h)
            
            # Display input data
            input_df = pd.DataFrame([{
                'G (kg/m2s)': G, 'x': x_val, 'Tsat (K)': T_input, 'rho_l': rho_l, 
                'rho_v': rho_v, 'mu_l': mu_l, 'mu_v': mu_v, 'k_v': k_v, 'k_l': k_l,
                'surface_tension (N/m)': surface_tension, 'Cp_v (J/kgK)': Cp_v, 'Cp_l (J/kgK)': Cp_l, 
                'Z': Z, 'Psat (Pa)': Psat, 'D (m)': D
            }])
            
            st.write("### Fluid Properties Used")
            st.dataframe(input_df)
            
            st.write(f"### <span style='color:blue;'>The predicted heat transfer coefficient is: **{predicted_h[0]:.4f} W/m²K**</span>", unsafe_allow_html=True)
            st.info("The Mean Absolute Percentage Error of the model is 9.22 %")
            
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

# Multiple Data Mode
elif mode == "Multiple Data":
    st.header("Multiple Data Processing")
    st.info("""
    Ensure your file includes all required fluid properties in the following order:
    Mass Flux (kg/m².s), Quality (x), Saturation Temperature (K), Density of liquid phase (kg/m³),
    Density of vapor phase (kg/m³), Dynamic viscosity of liquid phase (Ns/m²), Dynamic viscosity of vapor phase (Ns/m²),
    Thermal conductivity of vapor phase (W/m.K), Thermal conductivity of liquid phase (W/m.K),
    Surface Tension (N/m), Cp_v (J/kg.K), Cp_l (J/kg.K), Z parameter, Saturation pressure (Pa), Diameter (m)
    """)

    uploaded_file = st.file_uploader("Upload Excel or CSV file", type=["xlsx", "xls", "csv"])

    if uploaded_file is not None:
        try:
            # Define the exact feature names in the correct order
            feature_names = [
                'G (kg/m2s)', 'x', 'Tsat (K)', 'rho_l', 'rho_v', 'mu_l', 'mu_v', 
                'k_v', 'k_l', 'surface_tension', 'Cp_v', 'Cp_l', 'Z', 'Psat (Pa)', 'D (m)'
            ]
            
            # Read the file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, header=0)
            else:
                df = pd.read_excel(uploaded_file, engine='openpyxl', header=0)
            
            # Check if the file has the expected columns
            if list(df.columns) != feature_names:
                st.warning(f"Column names in file: {list(df.columns)}")
                st.warning(f"Expected column names: {feature_names}")
                st.info("Attempting to use columns in order...")
                # If columns don't match, assume they are in the right order but with different names
                df.columns = feature_names
            
            st.write("### Uploaded Data (first 5 rows):")
            st.dataframe(df.head())
            st.write(f"Data shape: {df.shape}")

            if st.button("Process Multiple Data"):
                epsilon = 1e-10
                
                # Apply log transform to all columns (keeping column names)
                df_log = df.copy()
                for col in df_log.columns:
                    df_log[col] = np.log(df_log[col] + epsilon)
                
                # Load PCA model
                pca = load_pca_model()
                
                # Transform using PCA
                X_pca = pca.transform(df_log)
                
                # Make predictions
                xgb_model = load_xgb_skopt_model()
                predicted_log_h = xgb_model.predict(X_pca)
                predicted_htc = np.exp(predicted_log_h)
                
                # Add predictions to original dataframe
                df['Predicted HTC (W/m²K)'] = predicted_htc

                st.write("### Processed Data (first 5 rows):")
                st.dataframe(df.head())
                
                # Show prediction statistics
                st.write(f"**Prediction Statistics:**")
                st.write(f"Min: {predicted_htc.min():.2f} W/m²K")
                st.write(f"Max: {predicted_htc.max():.2f} W/m²K")
                st.write(f"Mean: {predicted_htc.mean():.2f} W/m²K")
                st.write(f"Std: {predicted_htc.std():.2f} W/m²K")

                # Provide download option
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

                st.info("Processing complete!")
                st.session_state["df_processed"] = df

        except Exception as e:
            st.error(f"Error processing file: {e}")
            import traceback
            st.code(traceback.format_exc())

    # Graph Generation 
    if "df_processed" in st.session_state:
        st.write("### Generate Graph")
        processed_df = st.session_state["df_processed"]
        # Filter columns for graph selection (exclude non-numeric if any)
        numeric_cols = processed_df.select_dtypes(include=[np.number]).columns.tolist()
        x_var = st.selectbox("Select variable for X-axis", options=numeric_cols, key="x_axis")
        y_var = st.selectbox("Select variable for Y-axis", options=numeric_cols, key="y_axis")
        if st.button("Generate Graph"):
            fig, ax = plt.subplots()
            ax.scatter(processed_df[x_var], processed_df[y_var])
            ax.set_xlabel(x_var)
            ax.set_ylabel(y_var)
            ax.set_title(f"{y_var} vs {x_var}")
            st.pyplot(fig)
            buf = BytesIO()
            fig.savefig(buf, format="png", bbox_inches='tight', dpi=800)
            buf.seek(0)
            st.download_button(
                label="Download Graph as PNG",
                data=buf,
                file_name="graph.png",
                mime="image/png"
            )
