# app.py (Streamlit App)

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import shap
import os
import csv

st.set_page_config(page_title="Extrovert vs Introvert Classifier", layout="wide")

# Load model and scaler
model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")

# Sidebar input
st.sidebar.header("Enter Your Details")
time_alone = st.sidebar.slider("Time Spent Alone (hrs/day)", 0, 11, 4)
stage_fear = st.sidebar.radio("Stage Fear", ['Yes', 'No'])
social_events = st.sidebar.slider("Social Events per Week", 0, 10, 3)
going_out = st.sidebar.slider("Going Outside (days/week)", 0, 7, 4)
drained = st.sidebar.radio("Drained After Socializing", ['Yes', 'No'])
friends = st.sidebar.slider("Friends Circle Size", 0, 15, 5)
posts = st.sidebar.slider("Social Media Post Frequency", 0, 10, 4)

# User Data
user_data = pd.DataFrame([[
    time_alone,
    1 if stage_fear == 'Yes' else 0,
    social_events,
    going_out,
    1 if drained == 'Yes' else 0,
    friends,
    posts
]], columns=[
    'Time_spent_Alone', 'Stage_fear', 'Social_event_attendance', 'Going_outside',
    'Drained_after_socializing', 'Friends_circle_size', 'Post_frequency'])

if st.sidebar.button("Predict Personality"):
    user_scaled = scaler.transform(user_data)
    prediction = model.predict(user_scaled)[0]
    label = "Extrovert" if prediction == 1 else "Introvert"

    st.success(f"### Predicted Personality: {label}")

    # Probability
    proba = model.predict_proba(user_scaled)[0]
    st.info(f"Probability → Introvert: {proba[0]:.2f}, Extrovert: {proba[1]:.2f}")

    # SHAP Explainability
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(user_scaled)

    st.subheader("Why this prediction?")
    plt.figure()
    shap.summary_plot(shap_values, user_data, plot_type="bar", show=False)
    st.pyplot(plt)

    # Feedback Loop
    if st.button("I think this prediction is wrong"):
        st.warning("Thanks! We'll use this to improve the model.")
        feedback_row = list(user_data.iloc[0]) + [prediction]
        header = list(user_data.columns) + ["Model_Prediction"]

        file_exists = os.path.exists("feedback.csv")
        with open("feedback.csv", "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)
            writer.writerow(feedback_row)

# Dataset display and SHAP
st.title("Extrovert vs Introvert Classifier")
st.markdown("Predict personality based on daily behavioral patterns")
df = pd.read_csv('/Users/yashsmacbook/Downloads/personality_dataset.csv')
df.fillna(df.mean(numeric_only=True), inplace=True)
df['Stage_fear'] = df['Stage_fear'].map({'Yes': 1, 'No': 0})
df['Drained_after_socializing'] = df['Drained_after_socializing'].map({'Yes': 1, 'No': 0})
df['Personality'] = df['Personality'].map({'Extrovert': 1, 'Introvert': 0})

# Visual Tabs
tab1, tab2 = st.tabs(["Visualizations", "Feature Importance"])

with tab1:
    st.subheader("Distributions by Personality")
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    sns.kdeplot(data=df, x="Time_spent_Alone", hue="Personality", ax=axs[0,0])
    sns.kdeplot(data=df, x="Social_event_attendance", hue="Personality", ax=axs[0,1])
    sns.kdeplot(data=df, x="Friends_circle_size", hue="Personality", ax=axs[0,2])
    sns.kdeplot(data=df, x="Post_frequency", hue="Personality", ax=axs[1,0])
    sns.kdeplot(data=df, x="Going_outside", hue="Personality", ax=axs[1,1])
    sns.kdeplot(data=df, x="Drained_after_socializing", hue="Personality", ax=axs[1,2])
    st.pyplot(fig)

with tab2:
    st.subheader("Top Important Features")
    importances = model.feature_importances_
    feat_df = pd.DataFrame({
        'Feature': user_data.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    st.bar_chart(feat_df.set_index("Feature"))
