import pandas as pd
import numpy as np
import os
import pickle
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Load data function
def load_data(file):
    return pd.read_excel(file)

# Preprocess data function
def preprocess_data(df, task):
    df['material_ref'] = df['material_ref'].replace('00000', np.nan).fillna(df['material_ref'].mode()[0])
    df['selling_price'] = df['selling_price'].fillna(df['selling_price'].median())
    df['quantity tons'] = pd.to_numeric(df['quantity tons'], errors='coerce')
    df['thickness'] = pd.to_numeric(df['thickness'], errors='coerce')
    df['width'] = pd.to_numeric(df['width'], errors='coerce')
    df = df.dropna(subset=['quantity tons', 'thickness', 'width'])

    if 'selling_price' in df.columns:
        Q1 = df['selling_price'].quantile(0.25)
        Q3 = df['selling_price'].quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df['selling_price'] < (Q1 - 1.5 * IQR)) | (df['selling_price'] > (Q3 + 1.5 * IQR)))]

    if task == 'regression':
        df['selling_price'] = np.log1p(df['selling_price'])
        df['new_feature'] = df['quantity tons'] * df['thickness']
        return df[['selling_price', 'quantity tons', 'thickness', 'width', 'new_feature']], None

    if task == 'classification':
        df = df[df['status'].str.upper().isin(['WON', 'LOST'])]
        df['status'] = df['status'].str.upper().map({'WON': 1, 'LOST': 0})
        df['new_feature'] = df['quantity tons'] * df['thickness']
        return df[['status', 'quantity tons', 'thickness', 'width', 'new_feature']], None

# Train regression model
def train_regression_model(X_train, y_train):
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model

# Train classification model
def train_classification_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

# Evaluate regression model
def evaluate_regression_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return mse

# Evaluate classification model
def evaluate_classification_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, predictions, output_dict=True)
    auc = roc_auc_score(y_test, probas)
    return report, auc

# Save model function
def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

# Streamlit app
def streamlit_app():
    st.title("Industrial Copper Modeling")

    # Upload data
    uploaded_file = st.file_uploader("Upload your dataset", type=['xlsx'])

    if uploaded_file:
        try:
            data = load_data(uploaded_file)
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return

        if st.button("Run ML Model"):
            try:
                # Regression task
                regression_data, _ = preprocess_data(data.copy(), task="regression") # type: ignore
                X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
                    regression_data.drop('selling_price', axis=1), regression_data['selling_price'], test_size=0.2, random_state=42
                )
                regression_model = train_regression_model(X_train_r, y_train_r)
                save_model(regression_model, 'regression_model.pkl')
                mse = evaluate_regression_model(regression_model, X_test_r, y_test_r)
                st.write(f"Regression Mean Squared Error: {mse}")

                # Classification task
                classification_data, _ = preprocess_data(data.copy(), task="classification") # type: ignore
                X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
                    classification_data.drop('status', axis=1), classification_data['status'], test_size=0.2, random_state=42
                )
                classification_model = train_classification_model(X_train_c, y_train_c)
                save_model(classification_model, 'classification_model.pkl')
                report, auc = evaluate_classification_model(classification_model, X_test_c, y_test_c)
                st.write("Classification Report:")
                st.json(report)
                st.write(f"ROC-AUC Score: {auc}")
            except Exception as e:
                st.error(f"Error during ML Model execution: {e}")

    task = st.selectbox("Select Task", ["Regression", "Classification"])

    if task == "Regression":
        st.subheader("Selling Price Prediction")
        quantity_tons = st.number_input("Quantity (tons)", min_value=0.0)
        thickness = st.number_input("Thickness", min_value=0.0)
        width = st.number_input("Width", min_value=0.0)

        if st.button("Predict Selling Price"):
            try:
                input_data = pd.DataFrame({
                    'quantity tons': [quantity_tons],
                    'thickness': [thickness],
                    'width': [width],
                    'new_feature': [quantity_tons * thickness]
                })
                with open('regression_model.pkl', 'rb') as file:
                    regression_model = pickle.load(file)
                prediction = regression_model.predict(input_data)
                st.write(f"Predicted Selling Price: {np.expm1(prediction[0])}")
            except Exception as e:
                st.error(f"Error during prediction: {e}")

    if task == "Classification":
        st.subheader("Lead Classification")
        quantity_tons = st.number_input("Quantity (tons)", min_value=0.0)
        thickness = st.number_input("Thickness", min_value=0.0)
        width = st.number_input("Width", min_value=0.0)

        if st.button("Predict Status"):
            try:
                input_data = pd.DataFrame({
                    'quantity tons': [quantity_tons],
                    'thickness': [thickness],
                    'width': [width],
                    'new_feature': [quantity_tons * thickness]
                })
                with open('classification_model.pkl', 'rb') as file:
                    classification_model = pickle.load(file)
                prediction = classification_model.predict(input_data)
                status = "WON" if prediction[0] == 1 else "LOST"
                st.write(f"Predicted Status: {status}")
            except Exception as e:
                st.error(f"Error during prediction: {e}")

if __name__ == "__main__":
    streamlit_app()
