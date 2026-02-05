import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
df=pd.read_csv(r'c:\\Users\\DELL\\Downloads\\CAR DETAILS FROM CAR DEKHO.csv')
df.drop(['name','fuel','seller_type','transmission','owner'],axis=1,inplace=True)
df.drop_duplicates(inplace=True)
scaler=MinMaxScaler()
X=df.drop(['year'],axis=1)
y=df['year']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
lr=LinearRegression()
X_train_lr=lr.fit(X_train_scaled,y_train)

y_predic=lr.predict(X_test_scaled)

def predict_year(input_data):
    input_df=pd.DataFrame([input_data],columns=X.columns)
    input_scaled=scaler.transform(input_df)
    predict_year=lr.predict(input_scaled)
    return predict_year[0]



input_data={
    int(input("Enter Selling Price :")),
    int(input("Enter KM driven :"))
}

predicted_year=predict_year(input_data)
print(f"Predicted Year : {predicted_year}")