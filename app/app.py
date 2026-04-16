import streamlit as st
import joblib
import json
import numpy as np

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bengaluru Home Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0f0f0f;
    color: #f0ede6;
}

.main { background-color: #0f0f0f; }

h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem !important;
    color: #e8c97e !important;
    letter-spacing: -0.5px;
    line-height: 1.2;
}

.subtitle {
    color: #888;
    font-size: 1rem;
    font-weight: 300;
    margin-top: -12px;
    margin-bottom: 32px;
}

div[data-testid="stSelectbox"] label,
div[data-testid="stNumberInput"] label {
    color: #aaa !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 500;
}

div[data-testid="stSelectbox"] > div,
div[data-testid="stNumberInput"] > div > div {
    background-color: #1a1a1a !important;
    border: 1px solid #2e2e2e !important;
    border-radius: 8px !important;
    color: #f0ede6 !important;
}

div[data-testid="stSelectbox"] > div:focus-within,
div[data-testid="stNumberInput"] > div > div:focus-within {
    border-color: #e8c97e !important;
    box-shadow: 0 0 0 1px #e8c97e33 !important;
}

.stButton > button {
    background: linear-gradient(135deg, #e8c97e, #c9a84c);
    color: #0f0f0f;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    font-size: 0.95rem;
    letter-spacing: 0.5px;
    border: none;
    border-radius: 8px;
    padding: 0.65rem 2.5rem;
    width: 100%;
    cursor: pointer;
    transition: opacity 0.2s;
}

.stButton > button:hover { opacity: 0.88; }

.result-card {
    background: linear-gradient(135deg, #1a1a1a, #141414);
    border: 1px solid #e8c97e44;
    border-radius: 12px;
    padding: 28px 32px;
    text-align: center;
    margin-top: 24px;
}

.result-label {
    color: #888;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 8px;
}

.result-value {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    color: #e8c97e;
    line-height: 1;
}

.result-unit {
    color: #aaa;
    font-size: 0.9rem;
    margin-top: 6px;
}

.divider {
    border: none;
    border-top: 1px solid #2e2e2e;
    margin: 28px 0;
}
</style>
""", unsafe_allow_html=True)


# ── Load artifacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("banglore_home_prices_model.joblib")

@st.cache_data
def load_columns():
    with open("columns.json") as f:
        data = json.load(f)
    return data["data_columns"]

model = load_model()
all_columns = load_columns()

# Derive location list: every column that is NOT a base feature
base_features = {"total_sqft", "bath", "price", "bhk"}
locations = sorted([c for c in all_columns if c not in base_features])
locations = ["Others"] + locations   # 'Others' first


# ── Prediction helper ─────────────────────────────────────────────────────────
def predict_price(location: str, sqft: float, bath: int, bhk: int) -> float:
    x = np.zeros(len(all_columns))
    x[all_columns.index("total_sqft")] = sqft
    x[all_columns.index("bath")]       = bath
    x[all_columns.index("bhk")]        = bhk

    loc_key = location.lower()
    if loc_key in all_columns:
        x[all_columns.index(loc_key)] = 1   # one-hot

    return round(model.predict([x])[0], 2)


# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("<h1>Bengaluru Home<br>Price Predictor</h1>", unsafe_allow_html=True)
st.markdown('<p class="subtitle">Estimate property value in seconds</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    location = st.selectbox("Location", locations)
    sqft     = st.number_input("Total Sq. Ft.", min_value=100.0, max_value=50000.0,
                                value=1000.0, step=50.0)

with col2:
    bhk  = st.number_input("BHK", min_value=1, max_value=20, value=2, step=1)
    bath = st.number_input("Bathrooms", min_value=1, max_value=20, value=2, step=1)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

if st.button("Estimate Price"):
    price = predict_price(location, sqft, bath, bhk)
    st.markdown(f"""
    <div class="result-card">
        <div class="result-label">Estimated Price</div>
        <div class="result-value">₹{price}</div>
        <div class="result-unit">Lakhs &nbsp;·&nbsp; {location}</div>
    </div>
    """, unsafe_allow_html=True)