# Gender -> 1:Female 0:Male
# Churn -> 1:Yes 0:No

# What is StandardScaler?
# A tool from scikit-learn used for feature scaling.
# It standardizes numerical data so that:
# mean = 0 standard deviation = 1
# Many machine learning algorithms (like logistic regression, SVM, KNN, neural networks) work better when features are on the same scale.
# Prevents features with large values (e.g., salary in lakhs) from dominating small ones (e.g., age).

# scaler.fit_transform(X_train)
# fit() → calculates the mean and standard deviation of each column in X_train.
# transform() → uses those values to scale the data (so mean = 0, std = 1).

# joblib is a Python library used for saving and loading Python objects efficiently.
# joblib.dump(object, filename) → saves (serializes) a Python object to a file.

# Model is exported as model.pkl

# order of X: Age->Gender->Tenure->MonthlyCharges

import streamlit as st
import joblib
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- LOAD ARTIFACTS ---
# Note: Ensure these files are in the same directory as your app.py
try:
    scaler = joblib.load("scaler.pkl")
    model = joblib.load("model.pkl")
except FileNotFoundError:
    st.error("Error: `scaler.pkl` or `model.pkl` not found. Please ensure they are in the correct directory.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading model files: {e}")
    st.stop()


# --- CUSTOM CSS FOR A MODERN & SIMPLE UI ---
st.markdown("""
<style>
    /* Import a clean, professional font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    /* General body and container styling */
    body {
        font-family: 'Inter', sans-serif;
        background-color: #F0F2F6; /* Light grey background */
    }

    /* Main content block styling */
    .main .block-container {
        padding: 2rem;
        border-radius: 1rem;
        background-color: #FFFFFF;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    
    /* Title styling */
    h1 {
        color: #1a202c; /* Darker text for contrast */
        text-align: center;
        font-weight: 700;
    }
    
    /* Subheader/instruction styling */
    .st-emotion-cache-1629p8f p {
        text-align: center;
        color: #4a5568;
    }

    /* Input widgets styling */
    .stNumberInput > label, .stSelectbox > label {
        color: #D3D3D3 !important;
        font-weight: 600;
    }

    /* Button styling */
    .stButton > button {
        width: 100%;
        border-radius: 0.5rem;
        border: none;
        background-color: #2563eb; /* A modern blue */
        color: white;
        font-weight: 600;
        padding: 0.75rem 0;
        transition: background-color 0.2s ease-in-out, transform 0.1s ease-in-out;
    }
    .stButton > button:hover {
        background-color: #1d4ed8; /* Darker blue on hover */
    }
    .stButton > button:active {
        transform: scale(0.98);
    }

    /* Prediction result card styling */
    .churn-card {
        background-color: #f7fafc;
        border: 1px solid #e2e8f0;
        padding: 1.5rem;
        border-radius: 0.75rem;
        text-align: center;
        margin-top: 1.5rem;
    }
    .churn-card h3 {
        color: #1a202c;
        margin-bottom: 0.5rem;
        font-size: 1.25rem;
    }
    .churn-yes {
        font-size: 1.75rem;
        font-weight: 700;
        color: #e53e3e; /* A clear red */
    }
    .churn-no {
        font-size: 1.75rem;
        font-weight: 700;
        color: #38a169; /* A clear green */
    }
    .churn-card small {
        color: #718096;
    }
</style>
""", unsafe_allow_html=True)


# --- APP LAYOUT ---
st.title("Customer Churn Predictor 📊")
st.markdown('<p style="text-align: center; color: white;">Provide customer details below to predict their likelihood of churn.</p>', unsafe_allow_html=True)

st.divider()

# --- INPUT FORM ---
with st.form("churn_prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1, help="Enter the customer's age.")
        tenure = st.number_input("Tenure (Months)", min_value=0, max_value=150, value=12, step=1, help="How many months has the customer been with us?")

    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"], help="Select the customer's gender.")
        monthly_charge = st.number_input("Monthly Charge ($)", min_value=10.0, max_value=200.0, value=75.50, step=0.5, help="Enter the customer's monthly subscription fee.")

    # Submit button for the form
    submit_button = st.form_submit_button(label="Predict Churn")


# --- PREDICTION LOGIC ---
if submit_button:
    # 1. Preprocess the inputs
    gender_numeric = 1 if gender == "Female" else 0
    
    # 2. Create the feature vector
    features = np.array([[age, gender_numeric, tenure, monthly_charge]])
    
    # 3. Scale the features
    scaled_features = scaler.transform(features)
    
    # 4. Make a prediction
    prediction = model.predict(scaled_features)
    # This line has been kept from your original logic
    predicted = "Yes, Churn" if prediction == 1 else "No, Not Churn"

    # 5. Display the result
    st.divider()
    
    if prediction[0] == 1:
        churn_prob = predicted
        st.markdown(
            f"""
            <div class="churn-card">
                <h3>Prediction Result</h3>
                <p class="churn-yes">Likely to Churn</p>
                <p>Confidence: <strong>{churn_prob}%</strong></p>
                <small>Consider taking retention actions for this customer.</small>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        no_churn_prob = predicted
        st.markdown(
            f"""
            <div class="churn-card">
                <h3>Prediction Result</h3>
                <p class="churn-no">Unlikely to Churn</p>
                <p>Confidence: <strong>{no_churn_prob}%</strong></p>
                <small>This customer appears to be loyal.</small>
            </div>
            """,
            unsafe_allow_html=True
        )
else:
    st.info("Please fill in the details above and click 'Predict Churn' to see the result.")