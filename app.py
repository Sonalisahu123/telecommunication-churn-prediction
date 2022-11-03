import pandas as pd
import streamlit as st
import pickle
from pickle import dump
from pickle import load
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier


# load the model from disk
loaded_model = pickle.load(open('model_ensemble2.sav', 'rb'))


st.title('Model Deployment:Churn rate prediction')

st.sidebar.header('User Input Parameters')

def user_input_features():
    voice_mail_messages     = st.sidebar.number_input('Voice Mail Messages',min_value=0)
    evening_minutes         = st.sidebar.number_input('Evening Minutes')
    day_minutes             = st.sidebar.number_input('Day Minutes')
    customer_service_calls  = st.sidebar.number_input('Customer Service Calls', min_value=0)
    international_plan  = st.sidebar.radio('International Plan', ['Yes','No'])
    if international_plan == 'Yes':
        international_plan     = 1
    else:
        international_plan     = 0
    day_charge              = st.sidebar.number_input('Day Charge', min_value=0)
    total_charge            = st.sidebar.number_input('Total Charge')

    data = {'voice_mail_messages':voice_mail_messages ,'evening_minutes':evening_minutes  ,'day_minutes':day_minutes,
    'customer_service_calls':customer_service_calls,'international_plan':international_plan,'day_charge':day_charge,'total_charge':total_charge}  
 
    features = pd.DataFrame(data,index = [0])
    return features 
    
df = user_input_features()
st.subheader('User Input parameters')
st.write(df)




prediction = loaded_model.predict(df)


st.subheader('Predicted Result')
st.write('No,This person is not going to churn' if prediction==0 else ' Yes,This person is  going to churn')

