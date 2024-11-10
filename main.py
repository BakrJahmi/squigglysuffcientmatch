import streamlit as st
import pandas as pd
import pickle 
import numpy as np
import os
from openai import OpenAI
import plotly.graph_objects as go

# Set up OpenAI client
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY")
)

# Function to load machine learning models
def load_model(filename):
    try:
        with open(filename, "rb") as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.error(f"Model file '{filename}' not found.")
        return None
    except pickle.UnpicklingError:
        st.error(f"Error loading model from '{filename}'. File might be corrupted.")
        return None

# Load models
xgboost_model = load_model('xgb_model.pkl')
random_forest_model = load_model('rf_model.pkl')
knn_model = load_model('knn_model.pkl')

# Define helper functions for visualization
def create_gauge_chart(value):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        title={'text': "Churn Probability (%)"},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "darkblue"}}
    ))
    return fig

def create_model_probability_chart(probabilities):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(probabilities.keys()),
        y=list(probabilities.values()),
        text=[f"{v:.2%}" for v in probabilities.values()],
        textposition='auto'
    ))
    fig.update_layout(
        title="Model Probabilities",
        xaxis_title="Models",
        yaxis_title="Probability",
        yaxis_tickformat=".0%",
    )
    return fig

# Prepare input data for prediction
def prepare_input(credit_score, location, gender, age, tenure, balance,
      num_products, has_credit_card, is_active_member, estimated_salary):

    input_dict = {
        'CreditScore': credit_score,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'HasCrCard': int(has_credit_card),
        'IsActiveMember': int(is_active_member),
        'EstimatedSalary': estimated_salary, 
        'Geography_France': 1 if location == 'France' else 0,
        'Geography_Germany': 1 if location == 'Germany' else 0,
        'Geography_Spain': 1 if location == 'Spain' else 0,
        'Gender_Male': 1 if gender == 'Male' else 0,
        'Gender_Female': 1 if gender == 'Female' else 0
    }

    input_df = pd.DataFrame([input_dict])
    return input_df, input_dict

# Make predictions using loaded models
def make_predictions(input_df, input_dict):

    probabilities = {
        'XGBoost': xgboost_model.predict_proba(input_df)[0][1],
        'Random Forest': random_forest_model.predict_proba(input_df)[0][1],
        'K-Nearest Neighbors': knn_model.predict_proba(input_df)[0][1]
    }

    avg_probability = np.mean(list(probabilities.values()))

    st.markdown("### Model Probabilities")
    for model, prob in probabilities.items():
        st.write(f"{model}: {prob:.2%}")
    st.write(f"Average Probability: {avg_probability:.2%}")

    col1, col2 = st.columns(2)

    with col1:
        fig = create_gauge_chart(avg_probability)
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"The customer has a {avg_probability:.2%} probability of churning.")

    with col2:
        fig_probs = create_model_probability_chart(probabilities)
        st.plotly_chart(fig_probs, use_container_width=True)

    return avg_probability

# Explanation for churn prediction
def explain_prediction(probability, input_dict, surname):
    prompt = f"""You are an expert data scientist at a bank, where you specialize in 
interpreting and explaining predictions of machine learning models.

    Your machine learning model has predicted that a customer named {surname} has a 
{round(probability * 100, 1)}% probability of churning, based on the information provided below.

    Here is the customer's information:
    {input_dict}

    Feature importances:
    - NumOfProducts
    - IsActiveMember
    - Age
    - Geography_Germany
    - Balance
    - Geography_France
    - Gender_Female
    - Geography_Spain
    - CreditScore
    - EstimatedSalary
    - HasCrCard
    - Tenure
    - Gender_Male

    If the customer has over a 40% risk of churning, generate a 3 sentence explanation 
    of why they are at risk of churning. If the customer has less than a 40% risk of churning, generate a 3 sentence explanation of why they might not be at risk of churning.
    """

    raw_response = client.chat.completions.create(
        model="llama-3.2-3b-preview",
        messages=[{"role": "user", "content": prompt}]
    )
    return raw_response.choices[0].message.content

# Generate personalized email based on prediction
def generate_email(probability, input_dict, explanation, surname):
    prompt = f"""You are a manager at HS Bank responsible for customer retention.

    Customer info: {input_dict}
    Explanation of churn risk: {explanation}

    Write a friendly email to {surname}, offering incentives to stay. Use bullet points and avoid mentioning the churn model.
    """

    raw_response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    return raw_response.choices[0].message.content

# Streamlit application
st.title("Customer Churn Prediction")

# Load data
df = pd.read_csv("churn.csv")

# Create a list of customer options for the select box
customers = [f"{row['CustomerId']} - {row['Surname']}" for _, row in df.iterrows()]

# Customer selection
selected_customer_option = st.selectbox("Select a customer", customers)

if selected_customer_option:
    selected_customer_id = int(selected_customer_option.split(" - ")[0])
    selected_surname = selected_customer_option.split(" - ")[1]
    selected_customer = df.loc[df["CustomerId"] == selected_customer_id].iloc[0]

    # Display customer details
    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=int(selected_customer["CreditScore"]))
        location = st.selectbox("Location", ["Spain", "France", "Germany"], index=["Spain", "France", "Germany"].index(selected_customer['Geography']))
        gender = st.radio("Gender", ["Male", "Female"], index=0 if selected_customer['Gender'] == 'Male' else 1)
        age = st.number_input("Age", min_value=18, max_value=100, value=int(selected_customer['Age']))
        tenure = st.number_input("Tenure (years)", min_value=0, max_value=50, value=int(selected_customer['Tenure']))

    with col2:
        balance = st.number_input("Balance", min_value=0.0, value=float(selected_customer['Balance']))
        num_products = st.number_input("Number of Products", min_value=1, max_value=10, value=int(selected_customer['NumOfProducts']))
        has_credit_card = st.checkbox("Has Credit Card", value=bool(selected_customer['HasCrCard']))
        is_active_member = st.checkbox("Is Active Member", value=bool(selected_customer['IsActiveMember']))
        estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=float(selected_customer['EstimatedSalary']))

    input_df, input_dict = prepare_input(credit_score, location, gender, age, tenure, balance, num_products, has_credit_card, is_active_member, estimated_salary)
    avg_probability = make_predictions(input_df, input_dict)
    explanation = explain_prediction(avg_probability, input_dict, selected_customer['Surname'])

    st.markdown("---")
    st.subheader("Explanation of Prediction")
    st.markdown(explanation)

    email = generate_email(avg_probability, input_dict, explanation, selected_customer['Surname'])
    st.markdown('---')
    st.subheader("Personalized Email")
    st.markdown(email)
