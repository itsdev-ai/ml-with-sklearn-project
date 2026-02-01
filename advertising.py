import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,r2_score #--> iski help se hum pata laga sakte hai hamara predict output kitni sahi hai


df=pd.read_csv(r'C:\\Users\\DELL\\Desktop\\PROJECT\\Advertising.csv')
y=df['Sales']
df.drop(['Unnamed','Sales'],axis=1,inplace=True)
X_train,X_test,y_train,y_test=train_test_split(df.iloc[:,:],y.iloc[:],test_size=0.2,random_state=42)
lr=LinearRegression()
lr.fit(X_train,y_train)
try:
    y_predict=lr.predict(X_test)
    print('Model trained successfully')
except Exception as e:
    print('Error :',e)

# # check_error=mean_absolute_error(y_test,y_predict)
# # print(check_error)
# # print(r2_score(y_test,y_predict))

def predict_sales(tv_budget,radio_budget,newspaper_budget):
    features=np.array([[tv_budget,radio_budget,newspaper_budget]])
    result=lr.predict(features)
    return result[0]

sales=predict_sales(tv_budget=230.1,radio_budget=37.8,newspaper_budget=69.2)
print(sales)

pickle.dump(lr,open('Linear_regression_model.pkl','wb'))