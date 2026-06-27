import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from tabulate import tabulate
from sklearn.preprocessing import StandardScaler

df= pd.read_csv(r"C:\Users\nisar\Desktop\AI Ml\Data Processing\Machine-Learning-Part-1\Project_2\heart.csv")
print(df.head())
print(df.info())
print(df.isnull().sum())
print(df.describe())
print(df.columns)


# non_numerical=['Sex','ChestPainType','RestingECG','ExerciseAngina','ST_Slope']
# for col in non_numerical:
#     sns.countplot(x=df[col])
#     plt.show()

print(df.shape)

ch_mean=df.loc[df['Cholesterol']!=0,'Cholesterol'].mean()
df['Cholesterol']=df['Cholesterol'].replace(0,ch_mean)
numerical_value=['Age','RestingBP','Cholesterol','FastingBS','MaxHR','Oldpeak','HeartDisease']
for col in numerical_value:
    plt.figure(figsize=(8,6))
    sns.histplot(df[col],kde=True,bins=20)
    plt.show()


df_cl =df.copy()
# df_cl.drop_duplicates(inplace=True)
# print(df_cl.shape)

df_cl['Sex']=df_cl['Sex'].map({"M":1,"F":0})
print(df_cl.head())

categorical_columns = ['ChestPainType', 'RestingECG', 'ST_Slope']
df_cl = pd.get_dummies(df_cl, columns=categorical_columns, drop_first=True,dtype=int)


df_cl['ExerciseAngina']=df_cl['ExerciseAngina'].map({'N':0,'Y':1})

print(tabulate(df_cl.head(),headers='keys',tablefmt='grid'))

numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
scaler = StandardScaler()
df_cl[numerical_cols] = scaler.fit_transform(df_cl[numerical_cols])
print(df_cl.head())


