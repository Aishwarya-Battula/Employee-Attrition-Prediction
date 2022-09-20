import numpy as np
import pickle
import pandas as pd
import streamlit as st 

pickle_in = open("attr_predictor.pkl","rb")
classifier= pickle.load(pickle_in)

def welcome():
    return "Welcome All"
    
def attr_prediction(Age,Gender,MonthlyIncome,PerformanceRating,Total_Satisfaction):
    
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: Age
        in: query
        type: number
        required: true
      - name: Gender
        in: query
        type: number
        required: true
      - name: MonthlyIncome
        in: query
        type: number
        required: true
      - name: PerformanceRating
        in: query
        type: number
        required: true
      - name: Total_Satisfaction
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
    prediction=classifier.predict([[Age,Gender,MonthlyIncome,PerformanceRating,Total_Satisfaction]])
    print(prediction)
    return prediction
    
def main():
    st.title("Employee Attrition Predictor")
    html_temp = """
    <div style="background-color:blue;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Employee Attrition Predictor ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    Age =st.text_input("Age")
    Gender =st.text_input("Gender")
    MonthlyIncome =st.text_input("MonthlyIncome")
    PerformanceRating =st.text_input("PerformanceRating")
    Total_Satisfaction =st.text_input("Total_Satisfaction")
    result=""
    if st.button("Predict"):
        result=attr_prediction(Age,Gender,MonthlyIncome,PerformanceRating,Total_Satisfaction)
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")
        
if __name__=='__main__':
    main()
    
    
