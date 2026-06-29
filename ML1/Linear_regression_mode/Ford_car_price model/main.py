import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df =pd.read_csv(r"Machine_Learning\ML1\Linear_regression_mode\Ford_car_price model\ford.csv")
print(df.head())
print(df.shape)
print(df.dtypes)
print(df.columns)
print(df.isnull().sum())
df.drop_duplicates(inplace=True)

# numerical_values=['year','price','tax']
# for col in numerical_values:
#     plt.figure(figsize=(8,6))
#     sns.histplot(df[col],kde=True,bins=20)
#     plt.show()

# columns = df.columns
# for col in columns:
#     sns.countplot(x=df[col])
#     plt.show()

dumming =['transmission', 'fuelType','model' ]
df_cl =df.copy()
df_cl =pd.get_dummies(df_cl,columns=dumming,drop_first=True, dtype=int)

print(df_cl.info())

numerical_cols = ['year', 'mileage', 'tax', 'mpg', 'engineSize']
scaler = StandardScaler()
df_cl[numerical_cols] = scaler.fit_transform(df_cl[numerical_cols])

print(df_cl)
x=df_cl.drop(columns =['price'],axis =1)
y=df_cl['price']

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test =train_test_split(x,y,test_size =0.33,random_state=42)

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
model=LinearRegression()
model.fit(X_train,Y_train)
Y_pred=model.predict(X_test)
r2=r2_score(Y_test,Y_pred)
print(r2)