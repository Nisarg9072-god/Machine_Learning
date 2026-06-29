import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

file = pd.read_csv(r"Machine_Learning\ML1\Linear_regression_mode\Insurance\insurance.csv")
# print(file.head())
# print(file.shape)
# print(file.info())
# print(file.describe())
# print(file.isnull().sum())\
# print(file.columns)

# numeric_columns=['age','bmi', 'children', 'charges']
# for col in numeric_columns:
#     plt.figure(figsize=(6,4))
#     sns.histplot(file[col] , kde =True , bins=20)
#     plt.show()

# sns.countplot(x=file['sex'])
# plt.show()

# sns.countplot(x=file['region'])
# plt.show()

# sns.countplot(x=file['smoker'])
# plt.show()

# plt.figure(figsize=(8,6))
# sns.heatmap(file.corr(numeric_only=True),annot=True)
# plt.show()

file_cleaned =file.copy()
print(file_cleaned.head()) 

df_clean=file.copy()
print(df_clean.shape)

df_clean.drop_duplicates(inplace=True)
print(df_clean.shape)
print(df_clean.isnull().sum())
print(df_clean.dtypes)

print(df_clean['sex'].value_counts())
df_clean['sex']= df_clean['sex'].map({"male":0,"female":1})
print(df_clean.head())

print(df_clean['smoker'].value_counts())
df_clean['smoker']= df_clean['smoker'].map({"no":0,"yes":1})
print(df_clean.head())

df_clean.rename(
    columns={
        "sex":"is_female",
        "smoker": "is_smoker"
},inplace=True)

print(df_clean.head())

df_clean=pd.get_dummies(df_clean,columns=['region'],drop_first=True)
df_clean=df_clean.astype(int)
print(df_clean.head())


df_clean['bmi_cat']=pd.cut(
    df_clean['bmi'],
    bins=[0,18.5,24.9,29.9, float('inf')],
    labels=['underweight','normal','overweight','obese']
)
df_clean=pd.get_dummies(df_clean,columns=['bmi_cat'],drop_first=True)
df_clean=df_clean.astype(int)
cols =['age','bmi','children']
scaler= StandardScaler()
df_clean[cols]=scaler.fit_transform(df_clean[cols])


print(df_clean)

x=df_clean.drop('charges',axis=1)
y=df_clean['charges']
X_train,X_test ,y_train,y_test =train_test_split(x,y,test_size =0.20, random_state=42)

from sklearn.linear_model import LinearRegression

model=LinearRegression()
model.fit(X_train,y_train) 

y_pred =model.predict(X_test)

from sklearn.metrics import r2_score

r2=r2_score(y_test,y_pred)
print(r2)
n=X_test.shape[0]
p=X_test.shape[1]
adjusted_r2 =1-((1-r2)*(n-1)/(n-p-1))
print(adjusted_r2)