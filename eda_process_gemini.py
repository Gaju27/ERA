import os
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
import google.generativeai as genai

# ------------------------
# Setup Gemini
# ------------------------
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def ask_gemini(prompt: str) -> str:
    """Query Gemini for guidance."""
    model = genai.GenerativeModel("gemini-1.5-flash")  # free tier friendly
    response = model.generate_content(prompt)
    return response.text.strip()

def get_model_instance(model_name: str, problem: str):
    """Minimal mapping of model names to scikit-learn models."""
    name = model_name.lower()

    if problem == "classification":
        if "logistic" in name: return LogisticRegression(max_iter=1000)
        elif "svm" in name: return SVC()
        elif "forest" in name: return RandomForestClassifier()
    else:  # regression
        if "linear" in name: return LinearRegression()
        elif "svm" in name: return SVR()
        elif "forest" in name: return RandomForestRegressor()
    return None

def auto_preprocess(df):
    """Simple preprocessing (impute, encode, scale)."""
    df = df.copy()

    for col in df.columns:
        if df[col].dtype in ["int64", "float64"]:
            df[col] = SimpleImputer(strategy="mean").fit_transform(df[[col]])
        else:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    df[num_cols] = StandardScaler().fit_transform(df[num_cols])

    return df

# ------------------------
# Streamlit App
# ------------------------
st.title("🤖 AutoML with Gemini (AWS Ready)")

uploaded_file = st.file_uploader("📂 Upload your CSV dataset", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview", df.head())

    target_col = st.selectbox("🎯 Choose target column", df.columns)

    if target_col:
        X = df.dropna().drop(columns=[target_col])
        y = df.dropna()[target_col]

        # Detect classification vs regression
        problem = "classification" if (y.nunique() <= 20 and y.dtype != "float") else "regression"
        st.write(f"Detected problem type: **{problem}**")

        if st.button("🚀 Train"):
            # Preprocess
            df_clean = auto_preprocess(df)
            X = df_clean.drop(columns=[target_col])
            y = df_clean[target_col]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Ask Gemini for 3 candidate models
            model_prompt = f"Suggest 3 machine learning models suitable for {problem}."
            model_list_text = ask_gemini(model_prompt)
            st.info(f"🤖 Gemini suggests:\n{model_list_text}")

            model_names = [m.strip() for m in model_list_text.split("\n") if m.strip()]

            results = []

            for model_name in model_names:
                model = get_model_instance(model_name, problem)
                if model is None:
                    continue

                try:
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)

                    if problem == "classification":
                        score = accuracy_score(y_test, preds)
                        metric = f"Accuracy: {score:.2f}"
                    else:
                        score = r2_score(y_test, preds)
                        rmse = mean_squared_error(y_test, preds, squared=False)
                        metric = f"R²: {score:.2f}, RMSE: {rmse:.2f}"

                    results.append((model_name, metric, score))
                    st.success(f"✅ {model_name} → {metric}")

                except Exception as e:
                    st.error(f"⚠️ {model_name} failed: {e}")

            # Pick the best model
            if results:
                best = max(results, key=lambda x: x[2])
                st.balloons()
                st.subheader(f"🏆 Best Model: {best[0]} → {best[1]}")
