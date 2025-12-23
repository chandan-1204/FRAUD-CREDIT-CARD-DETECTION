import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go

def load_css():
    st.markdown("""
    <style>
    /* Background */
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: white;
        animation: fadeIn 1.2s ease-in;
    }

    /* Fade animation */
    @keyframes fadeIn {
        from {opacity: 0;}
        to {opacity: 1;}
    }

    /* Main title */
    h1 {
        text-shadow: 0 0 15px rgba(0, 255, 255, 0.6);
        animation: glow 2s infinite alternate;
    }

    @keyframes glow {
        from { text-shadow: 0 0 10px #00f2ff; }
        to { text-shadow: 0 0 25px #00f2ff; }
    }

    /* Input cards */
    div[data-testid="stNumberInput"] {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 10px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    div[data-testid="stNumberInput"]:hover {
        transform: scale(1.02);
        box-shadow: 0 0 15px rgba(0, 255, 255, 0.3);
    }

    /* Predict button */
    div.stButton > button {
        background: linear-gradient(90deg, #ff416c, #ff4b2b);
        color: white;
        border-radius: 30px;
        padding: 0.6em 1.5em;
        font-size: 16px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    div.stButton > button:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 0 20px rgba(255, 75, 43, 0.7);
    }

    /* Result alerts */
    .stAlert {
        border-radius: 15px;
        animation: slideUp 0.6s ease-out;
    }

    @keyframes slideUp {
        from { transform: translateY(20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    </style>
    """, unsafe_allow_html=True)

# Load model and scaler
model = joblib.load("models/fraud_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Page config
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="centered"
)
load_css()

# Title
st.markdown("<h1 style='text-align:center;'>🛡️ Credit Card Fraud Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>AI-powered fraud detection using Machine Learning</p>", unsafe_allow_html=True)
st.markdown("---")

# Input Section
st.subheader("🔢 Enter Transaction Details")

time = st.number_input("Transaction Time (seconds)", min_value=0.0, value=1000.0)
amount = st.number_input("Transaction Amount", min_value=0.0, value=1000.0)

st.info("ℹ️ PCA-based transaction features are generated automatically for prediction.")

# Predict Button
if st.toast("Prediction completed successfully 🚀"):

    # Generate realistic PCA values (V1–V28)
    np.random.seed(42)
    pca_features = np.random.normal(loc=0, scale=1, size=28)

    # Scale amount
    amount_scaled = scaler.transform([[amount]])[0][0]

    # Final input vector
    input_data = np.array([[time] + list(pca_features) + [amount_scaled]])

    # Prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.markdown("---")

    # Result Display
    if prediction == 1:
        st.error("🚨 FRAUDULENT TRANSACTION DETECTED")
    else:
        st.success("✅ NORMAL TRANSACTION")

    st.write(f"**Fraud Probability:** {probability:.2f}")

    # Gauge Chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={'text': "Fraud Risk (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "red"},
            'steps': [
                {'range': [0, 40], 'color': "lightgreen"},
                {'range': [40, 70], 'color': "orange"},
                {'range': [70, 100], 'color': "red"}
            ],
        }
    ))

    st.plotly_chart(fig, use_container_width=True)
