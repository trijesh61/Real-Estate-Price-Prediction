# 🏠 Bengaluru House Price Prediction

A machine learning web application that predicts residential property prices in Bengaluru based on location, size, and amenities.

🔗 **Live App** → [bengaluru-house-price-prediction1.streamlit.app](https://bengaluru-house-price-prediction1.streamlit.app/)

---

## 📌 Overview

This project uses a **Linear Regression** model trained on the Bengaluru House Price dataset to estimate property values in Lakhs (₹). The app provides an interactive UI where users can select a location and input property details to get an instant price estimate.

---

## 🧠 ML Pipeline

- **Data Preprocessing** — Handled missing values, outliers, and feature engineering (BHK extraction, price per sqft)
- **Feature Engineering** — One-hot encoding for 200+ Bengaluru locations
- **Model** — Linear Regression with `GridSearchCV` for hyperparameter tuning
- **Evaluation** — Cross-validated using `ShuffleSplit` (5 folds, 80/20 split)
- **Serialization** — Model saved with `joblib`, column metadata saved as `columns.json`

---

## 🗂️ Project Structure

```
real-estate-price-prediction/
└── app/
    ├── app.py                              # Streamlit application
    ├── banglore_home_prices_model.joblib   # Trained ML model
    ├── columns.json                        # Feature column metadata
    └── requirements.txt                   # Python dependencies
```

---

## 🚀 Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/<your-username>/real-estate-price-prediction.git
cd real-estate-price-prediction/app
```

**2. Install dependencies**
```bash
py -3.11 -m pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

---

## 📦 Dependencies

| Package | Version |
|---|---|
| streamlit | 1.54.0 |
| scikit-learn | 1.8.0 |
| pandas | 2.3.3 |
| numpy | 1.26.4 |
| joblib | 1.5.3 |

---

## 🖥️ App Features

- **Location dropdown** — 200+ Bengaluru localities + *Others* category
- **Input fields** — Total Sq. Ft., BHK, and Bathrooms
- **Instant prediction** — Estimated price displayed in ₹ Lakhs
- **Deployed on Streamlit Cloud** — No setup required to try it

---

## 📊 Dataset

- **Source** — [Bengaluru House Price Data](https://www.kaggle.com/amitabhajoy/bengaluru-house-price-data) (Kaggle)
- **Records** — ~13,000 property listings
- **Features** — Location, Size, Total Sqft, Bath, Price

---

## 👤 Author

Built with Python, scikit-learn, and Streamlit.  
Feel free to fork, star ⭐, and contribute!
