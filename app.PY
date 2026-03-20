import streamlit as st
import pandas as pd
import joblib

st.title("HEALTH CARE RISK MANAGEMENT APP")

model = joblib.load('risk_model.pkl')

age = st.number_input("Age", min_value=0)
Length_of_stay = st.number_input("Length of stay(days)", min_value=0)
treatment_cost = st.number_input("Treatment Cost", min_value=0.0)
AbnormalLABCount = st.number_input("AbnormalLABCount", min_value=0)

if st.button("Predict", key="predict_btn"):
    
    input_data = pd.DataFrame(
        [[age, Length_of_stay, treatment_cost, AbnormalLABCount]],
        columns=['AGE','lengthofstay','TREAMENT COST','AbnormalLABCount']
    )
    
    probability = model.predict_proba(input_data)[0][1]
    prediction = 1 if probability > 0.7 else 0
    
    if prediction == 1:
        st.error("⚠️ High Risk Patient")
    else:
        st.success("✅ Low Risk Patient")
    
    st.write(f"Risk Probability: {probability:.2%}")





    
 