# 🧠 Extrovert vs Introvert Classifier

This project predicts whether a person is an **Extrovert** or **Introvert** based on their daily behavioral traits using machine learning and SHAP explainability.

Built with:

- XGBoost Classifier
- Streamlit for interactive dashboard
- SHAP for model explainability
- Feedback loop for continuous improvement

---

## 🚀 Features

- Predict personality using inputs like social media activity, time spent alone, and more
- View probability scores (Introvert vs Extrovert)
- Explainable predictions using SHAP
- Explore visualizations comparing introvert vs extrovert behavior
- Submit feedback if the prediction seems wrong

---

## 🧰 Tech Stack

- Python, Pandas, Scikit-learn
- XGBoost
- Streamlit
- SHAP
- Matplotlib, Seaborn

---

## 🧪 How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git
cd YOUR-REPO-NAME
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install streamlit pandas numpy xgboost shap seaborn matplotlib scikit-learn joblib
```

### 3. Run the Streamlit App

```bash
streamlit run app.py
```

### 4. Train or Retrain the Model (Optional)

Open and run the `personality_model.ipynb` notebook to:

- Train the model
- Save `xgb_model.pkl` and `scaler.pkl`
- Optionally incorporate feedback from `feedback.csv`

---

## 📁 Files Overview

- `app.py` – Streamlit app with prediction, SHAP plots, feedback loop
- `personality_model.ipynb` – Model training and evaluation
- `xgb_model.pkl`, `scaler.pkl` – Saved model and scaler
- `feedback.csv` – Logged user corrections
- `personality_dataset.csv` – Input dataset

---

## 📈 Example

![Screenshot](screenshot.png) _(<img width="1280" height="737" alt="image" src="https://github.com/user-attachments/assets/bb915347-824a-4f10-b83c-12439c235b22" />
)_

---

## 🤝 Contributing

Pull requests and improvements welcome!

---

## 📄 License

This project is for educational use only.
