import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings("ignore")

df=sns.load_dataset("titanic")
# print(df.head())
# print(df.shape)
# print(df.info())
# print(df.isnull().sum())
# print(df.describe())

df.drop(['deck','embark_town','alive','class','who','adult_male'],axis=1,inplace=True)

df['age'].fillna(df['age'].mean(),inplace=True)
df.dropna(subset=['embarked'],inplace =True)
# print(df.shape)
# print(df.info())

from sklearn.preprocessing import LabelEncoder
le =LabelEncoder()
df['sex']=le.fit_transform(df['sex'])
df['embarked']=le.fit_transform(df['embarked'])
df =df.astype(int)

from sklearn.model_selection import train_test_split
x=df.drop('survived',axis=1)
y=df['survived']
X_train,X_test, Y_train,Y_test =train_test_split(x,y,test_size=.2,random_state=42)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled,Y_train)

Y_pred = model.predict(X_test_scaled)
# print(Y_pred)
# print("\n")
# print(Y_test)
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print(accuracy_score (Y_test,Y_pred))
confusion_matrix(Y_test,Y_pred)
print(classification_report(Y_test,Y_pred))