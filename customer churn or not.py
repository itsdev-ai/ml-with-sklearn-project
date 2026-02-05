import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
df=pd.read_csv(r'c:\\Users\\DELL\\Downloads\\7 churn.csv')
df.drop(['customerID','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling'],axis=1,inplace=True)
df['TotalCharges']=pd.to_numeric(df['TotalCharges'],errors='coerce')
df.fillna(df['TotalCharges'].mean(),inplace=True)                   
le_dict = {}
lgr=LogisticRegression()
for col in ['gender','Contract','PaymentMethod','Partner','Dependents','PhoneService','Churn']:
    le_dict[col]=LabelEncoder()
    df[col]=le_dict[col].fit_transform(df[col])
y=df['Churn']
X=df.drop(['Churn'],axis=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
X_train_scaled_lgr=lgr.fit(X_train_scaled,y_train)
y_predict_lgr=lgr.predict(X_test_scaled)
try:
    def predictive(input_dict):
        input_df=pd.DataFrame([input_dict])
             
        for col in ['gender','Contract','PaymentMethod','Partner','Dependents','PhoneService']:
            input_df[col] = le_dict[col].transform([input_df[col].values[0]])[0]

        input_df_scaled = scaler.transform(input_df)
        churn_predict=lgr.predict(input_df_scaled).reshape(1,-1)
        return churn_predict[0]
    

    input_dict={

    'gender':'Male', 
        'SeniorCitizen':1, 
        'Partner':'No', 
        'Dependents':'No', 
        'tenure':1, 
        'PhoneService':'Yes', 
        'Contract':'Month-to-month', 
        'PaymentMethod':'Electronic check', 
        'MonthlyCharges':100.50, 
        'TotalCharges':10000.50
    # 'gender': input("Enter Gender (Male/Female): "),
    # 'SeniorCitizen': input("Enter Senior Citizen (0/1): "),
    # 'Partner': input("Enter Partner (Yes/No): "),
    # 'Dependents': input("Enter Dependents (Yes/No): "),
    # 'tenure': int(input("Enter tenure: ")),
    # 'PhoneService': input("Enter Phone Service (Yes/No): "),
    # 'Contract': input("Enter Contract (Month-to-month/One year/Two year): "),
    # 'PaymentMethod': input("Enter Payment Method: "),
    # 'MonthlyCharges': float(input("Enter Monthly Charges: ")),
    # 'TotalCharges': float(input("Enter Total Charges: "))
    }

    churn_predict=predictive(input_dict)
    print("It's work successfully")
    if churn_predict==0:
        print('Customer Churn : Not Churn')
    if churn_predict==1:
        print("Customer Churn : Churn")
except Exception as e:
    print("Error !!",e)
