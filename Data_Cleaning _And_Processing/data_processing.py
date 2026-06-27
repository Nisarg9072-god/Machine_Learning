import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler , LabelEncoder
from sklearn.model_selection import train_test_split
#create the sample data set
data = {
    'Name' : ['nisarg', 'ruchit', 'rahul', 'raj'],
    'age' : [19,np.nan,18,22],
    'gender':['male','male','male','male'],
    'Salary':[2000,23000,np.nan,3500]
}

df = pd.DataFrame(data)
print(df)

# handling the missing data

df['age'] =df['age'].fillna(df['age'].mean())
df['Salary']=df['Salary'].fillna(df['Salary'].mean())
print('\n')
print(df)

# Encode the categorical data
# convert male 0 and female 1

le = LabelEncoder()
df['gender']=le.fit_transform(df['gender'])
print('\n')
print(df)

# Feature Scaling
# convert the large data in to the smaller data so the mean deaviation and the standard deviation should be zero or one
# This will scale the age and salary in such a way that the mean is 0 and the standard deviation is 1

scaler = StandardScaler()
df[['age','Salary']] = scaler.fit_transform(df[['age','Salary']])
print('\n')
print(df)

#splitting the data set

x=df[['age','gender','Salary']]
y= [1,0,1,0]

x_train,x_test, y_train ,y_test= train_test_split(x, y, test_size=0.4,random_state=40)
print("\ntraining set:")
print(x_train)
print('\n test set:')
print(x_test)
