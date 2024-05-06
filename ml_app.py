# Core Package
import streamlit as st 

# import ML Packages
import joblib
import os

# Load EDA Packages
import pandas as pd 
import numpy as np 


attrib_info = """
#### Attribute Information:
    - Age 1.20-65
    - Sex 1. Male, 2.Female
    - Polyuria 1.Yes, 2.No.
    - Polydipsia 1.Yes, 2.No.
    - sudden weight loss 1.Yes, 2.No.
    - weakness 1.Yes, 2.No.
    - Polyphagia 1.Yes, 2.No.
    - Genital thrush 1.Yes, 2.No.
    - visual blurring 1.Yes, 2.No.
    - Itching 1.Yes, 2.No.
    - Irritability 1.Yes, 2.No.
    - delayed healing 1.Yes, 2.No.
    - partial paresis 1.Yes, 2.No.
    - muscle stiness 1.Yes, 2.No.
    - Alopecia 1.Yes, 2.No.
    - Obesity 1.Yes, 2.No.
    - Class 1.Positive, 2.Negative.
"""

label_dict = {"No" : 0, "Yes" : 1}
gender_map ={"Female" : 0, "Male" : 1}
target_label_map = {"Negative" : 0, "Positive" : 1}

def get_fvalue(val):
    feature_dict = {"No" : 0, "Yes" : 1}
    for key,value in feature_dict.items():
        if val == key:
            return value
        
        
def get_value(val, my_dict):
    for key,value in my_dict.items():
        if val == key:
            return value
        
        
# Load ML Models
@st.cache
def load_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file), "rb"))
    return loaded_model


def run_ml_app():
    st.subheader("ML Prediction")
    #st.write("It is working")
    #st.success("Cool")
    
    
    with st.expander("Attribute Information"):
        st.markdown(attrib_info)
        
        
    # Layout
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", 10, 100)
        gender = st.radio("Gender", ["Female", "Male"])
        polyuria = st.radio("Polyuria", ["No", "Yes"])
        polydipsia = st.radio("Polydipsia", ["No", "Yes"])
        sudden_weight_loss = st.selectbox("Sudden Weight_loss", ["No", "Yes"])
        weakness = st.radio("Weakness", ["No", "Yes"])
        polyphagia = st.radio("Polyphagia", ["No", "Yes"])
        genital_thrush = st.selectbox("Genital Thrush", ["No", "Yes"])
        
        
    with col2:
        visual_blurring = st.selectbox("Visual Blurring", ["No", "Yes"])
        itching = st.radio("Itching", ["No", "Yes"])
        irritability = st.radio("Irritability", ["No", "Yes"])
        delayed_healing = st.radio("Delayed Healing", ["No", "Yes"])
        partial_paresisi = st.selectbox("Partial Paresis", ["No", "Yes"])
        muscle_stiffness = st.radio("Muscle Stiffness", ["No", "Yes"])
        alopecia = st.radio("Alopecia", ["No", "Yes"])
        obesity = st.selectbox("Obesity", ["No", "Yes"])
        
        
    with st.expander("Your Selected Options"):
        result = {
            'Age' : age,
            'Gender' : gender,
            'Polyuria' : polyuria,
            'Polydipsia' : polydipsia,
            'Sudden Weight Loss' : sudden_weight_loss,
            'Weakness' : weakness,
            'Polyphagia' : polyphagia,
            'Genital Thrush' : genital_thrush,
            'Visual Blurring' : visual_blurring,
            'Itching' : itching,
            'Irritability' : irritability,
            'Delayed Healing' : delayed_healing,
            'Partial Paresisi' : partial_paresisi,
            'Muscle Stiffness' : muscle_stiffness,
            'Alopecia' : alopecia,
            'Obesity' : obesity
        }
        st.write(result)
        
        
        encoded_result = []
        for i in result.values():
            if type(i) == int:
                encoded_result.append(i)
            
            elif i in ["Female", "Male"]:
                res = get_value(i, gender_map)
                encoded_result.append(res)
                
            else:
                encoded_result.append(get_fvalue(i))
        
        st.write(encoded_result)
        
        
    with st.expander("Prediction Result"):
        single_sample = np.array(encoded_result).reshape(1, -1)
        st.write(single_sample)
        
        model = load_model("C:/Users/arya.hisma/Downloads/Small Project/07. Machine Learning/models/logistic_regression_model_diabetes_21_oct_2020.pkl")
        prediction = model.predict(single_sample)
        pred_prob = model.predict_proba(single_sample)
        st.write(prediction)
        st.write(pred_prob)
        
        if prediction == 1:
            st.warning("Positive Risk {}".format(prediction[0]))
            pred_probability_score = {
                "Negative DM Risk" : pred_prob[0][0] * 100,
                "Positive DM Risk" : pred_prob[0][1] * 100
                }
            st.dataframe(pred_probability_score, use_container_width=True)
            
            
        else:
            st.success("Negative Risk {}".format(prediction[0]))
            pred_probability_score = {
                "Negative DM Risk" : pred_prob[0][0] * 100,
                "Positive DM Risk" : pred_prob[0][1] * 100
                }
            st.dataframe(pred_probability_score, use_container_width=True)
            
        
        
    
        
        





























































