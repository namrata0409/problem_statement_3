import streamlit as st
import pickle
model = pickle.load(open('sol3.pkl', 'rb'))
def predict(TotalCharges, MonthlyCharges, SeniorCitizen, Tenure, Gender, Partner, Dependents, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod ):
    prediction = model.predict([[TotalCharges,MonthlyCharges,SeniorCitizen,Tenure, Gender, Partner, Dependents, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod]])
    return prediction
def main():
    st.title("Customer Churn Prediction")
    html_temp = """ <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Customer Churn Prediction </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    TotalCharges = st.text_input("Total Charges","Type Here")
    MonthlyCharges = st.text_input("Monthly Charges","Type Here")
    SeniorCitizen = st.text_input("Senior Citizen","Type Here")
    Tenure = st.text_input("Tenure","Type Here")
    Gender = st.text_input("Gender","Type Here")
    Partner = st.text_input("Partner","Type Here")
    Dependents = st.text_input("Dependents","Type Here")
    PhoneService = st.text_input("Phone Service","Type Here")
    MultipleLines = st.text_input("Multiple Lines","Type Here")
    InternetService = st.text_input("Internet Service","Type Here")
    OnlineSecurity = st.text_input("Online Security","Type Here")
    OnlineBackup = st.text_input("Online Backup","Type Here")
    DeviceProtection = st.text_input("Device Protection","Type Here")
    TechSupport = st.text_input("Tech Support","Type Here")
    StreamingTV = st.text_input("Streaming TV","Type Here")
    StreamingMovies = st.text_input("Streaming Movies","Type Here")
    Contract = st.text_input("Contract","Type Here")
    PaperlessBilling = st.text_input("Paperless Billing","Type Here")
    PaymentMethod = st.text_input("Payment Method","Type Here")

    result=""
    if st.button("Predict"):
        result= predict(TotalCharges, MonthlyCharges, SeniorCitizen, Tenure, Gender, Partner, Dependents, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod)
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Problem Statement 3")
main()
