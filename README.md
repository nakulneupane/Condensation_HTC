---

# 📘 Condensation Heat Transfer Coefficient Predictor

### Powered by XGBoost, PCA, CoolProp, and Gemini AI

---

## 📄 Overview

The **Condensation Heat Transfer Coefficient Predictor** is an interactive web application built using **Streamlit**. It enables researchers, engineers, and thermal scientists to:

* Predict condensation heat transfer coefficients (HTC) for a wide range of refrigerants and fluid combinations.
* Input and calculate thermophysical properties using **CoolProp** or provide them manually.
* Leverage advanced **machine learning** (XGBoost + Optuna + PCA) models trained on empirical datasets.
* Upload bulk data for batch processing.
* Interact with a **Google Gemini AI Assistant** for contextual help.
* Toggle between **light** and **dark UI themes**.

---

## 🧠 Core Features

### ✅ 1. HTC Prediction (Single & Multiple Data Points)

* Predicts the HTC using a trained XGBoost model with PCA for dimensionality reduction.
* Offers **manual input** or **automatic property calculation** (via CoolProp).

### ✅ 2. Fluid Property Handling

* Calculate key properties like density, viscosity, thermal conductivity, surface tension, etc.
* Supports **binary mixtures** with mass fractions using CoolProp’s `HEOS` backend.

### ✅ 3. Batch Processing (Multiple Data Mode)

* Upload Excel or CSV files for HTC prediction over multiple data entries.
* Returns a downloadable Excel file with HTC predictions appended.

### ✅ 4. Visualization

* Generate and download **scatter plots** based on predicted or uploaded data.

### ✅ 5. AI Assistant (Optional)

* Powered by **Google Gemini** (`gemini-2.0-flash` model).
* Users can ask questions related to condensation or thermophysical concepts.

### ✅ 6. Dynamic Theming

* Light/Dark mode toggle for improved user experience.

---

## 📌 Model Training & Validity

### 🔍 Validity Info

The XGBoost model was trained on data with the following ranges:

* **Mass Flux (G):** 24–1100 kg/m²s
* **Quality (x):** 0.01–0.99
* **Saturation Temperature:** 242–356 K
* **Inner Tube Diameter:** 0.49–20 mm
* **Refrigerants Supported:** 35+ types including R134a, R1234YF, R744, R717, binary blends like R290/R170, and more.

### 🧪 Accuracy

* **MAPE (Mean Absolute Percentage Error):** \~9.22%

---

## 🚀 How to Use the App

### 🔹 Initial Setup

1. Deploy the app locally or via **Streamlit Cloud**.
2. Add your **Google Gemini API key** to `.streamlit/secrets.toml` like so:

   ```toml
   GEMINI_API_KEY = "your_google_api_key_here"
   ```

### 🔹 Launch Modes

* **Single Data Point**
  Input all thermophysical values for one fluid condition.

* **Multiple Data**
  Upload `.xlsx` or `.csv` file with bulk entries for batch prediction.

### 🔹 Options for Fluid Property Input

* **CoolProp-based**: Auto-calculate using pressure/temperature + quality.
* **Manual**: Direct user input for all properties.

---

## 📁 Input Requirements

### For Multiple Data Upload:

The input file must include 14 columns in the following order:

1. Mass Flux (kg/m²s)
2. Quality (x)
3. Saturation Temperature (K)
4. Liquid Density (kg/m³)
5. Vapor Density (kg/m³)
6. Liquid Viscosity (Pa·s)
7. Vapor Viscosity (Pa·s)
8. Vapor Thermal Conductivity (W/m·K)
9. Liquid Thermal Conductivity (W/m·K)
10. Surface Tension (N/m)
11. Vapor Cp (J/kg·K)
12. Liquid Cp (J/kg·K)
13. Saturation Pressure (Pa)
14. Inner Tube Diameter (m)

---

## 📤 Outputs

### Single Mode

* Predicted HTC displayed in the app.
* Fluid property table shown for reference.

### Multiple Mode

* Predicted HTCs appended to uploaded dataset.
* Downloadable processed Excel file.

---

## 🛠 Technologies Used

| Component        | Tool/Library                  |
| ---------------- | ----------------------------- |
| Web UI           | Streamlit                     |
| ML Model         | XGBoost + Optuna + PCA        |
| Fluid Properties | CoolProp                      |
| Assistant AI     | LangChain + Google Gemini API |
| Plotting         | Matplotlib                    |
| File Handling    | pandas, joblib, BytesIO       |

---

## 🔐 Security & API Usage

* The app uses a **secure API key** stored in `.streamlit/secrets.toml`.
* Google Gemini API is only initialized when the key is valid.
* User-uploaded data is processed in-memory and not stored permanently.

---

## ⚠️ Limitations

* The model is **only valid within the training data bounds**.
* Gemini AI usage may incur **API quota limits** or require a Google Cloud account.
* The assistant will be disabled if no API key is found or if the model fails to load.

---

## 📬 Support & Contact

If you encounter issues or have questions, contact the developer at:

**Email**: \[[ge22m027@smail.iitm.ac.in](mailto:your-email@example.com)]
**GitHub**: \[github.com/nakulneupane]
**License**: IIT Madras

---


