# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 17:20:38 2021

@author: khurram
"""

import pandas as pd
import numpy as np
data=pd.read_csv('heart_disease.csv',names=range(0,14))
data.rename(columns={0:'age',1:'sex',2:'cp',3:'trestbps',4:'chol',5:'fbs',6:'restecg',7:'thalach',8:'exang',9:'oldpeak',10:'slope',11:'ca',12:'thal',13:'target'},inplace=True)
data = data.replace({'?': np.nan}).dropna().astype(float)
# converting the 123 to 1
data['target']=data['target'].replace(2,1)
data['target']=data['target'].replace(3,1)
data['target']=data['target'].replace(4,1)

# lets sepearate input output columns
df_x=data.drop(columns=['target']) # Input variable.
y=pd.DataFrame(data['target']) #Target Variable.

# lets apply log transformation and treat the skewd data
for col in df_x.columns:
    if df_x.skew().loc[col]>0.55:
        df_x[col]=np.log1p(df_x[col])
        
#lets remove the outliers using zscore
from scipy.stats import zscore
z=abs(zscore(df_x))
print(df_x.shape)
new=df_x.loc[(z<3).all(axis=1)]
print(new.shape)
# we can obsere some of our data is been reduced hence the outliers is been removed

#Before moving forward lets scale our data.
# lets scale the input variable
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(df_x)
x=pd.DataFrame(x,columns=df_x.columns)

from sklearn.model_selection import train_test_split,cross_val_score
import warnings
warnings.filterwarnings('ignore')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state=42,stratify=y)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,roc_curve,auc,confusion_matrix,classification_report
from sklearn.model_selection import cross_val_score,cross_val_predict

# finalising the Random Forest Model.
rfc1=RandomForestClassifier(random_state=48,n_estimators=500,criterion='gini')
rfc1.fit(x_train,y_train)
pred=rfc1.predict(x_test)
print("Accuracy for Random Forest classifier: ",accuracy_score(y_test,pred)*100)
print('Cross validation score with Kneighbors :',cross_val_score(rfc1,x,y,cv=5,scoring='accuracy').mean()*100)

# lets Save the model
import pickle
pickle.dump(rfc1,open('Heart_disease.pkl','wb'))





















