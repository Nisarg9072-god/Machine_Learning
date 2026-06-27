# import Iris flower dataset
from sklearn.datasets import load_iris;

#split the dataset into train and test
from sklearn.model_selection import train_test_split;

#machine learning algorithm used for classification task
from sklearn.tree import DecisionTreeClassifier;
from sklearn.linear_model import LinearRegression;

#to evaluate its performance
from sklearn.metrics import accuracy_score;

#to organize and display the dataset in a readable table format
import pandas as pd;

iris=load_iris()

x=iris.data
y=iris.target

print(f'Features Name:{iris.feature_names}')
print(f'Targets label name: {iris.target_names}')

#optional:view data
df=pd.DataFrame(x,columns=iris.feature_names)

#the output labels are mapped into three numericals values such as 0,1,2
df['species']=y

# to view the numerical values as targer names we need to map them accordingly
# df['species'] = pd.seriers(y).map(dict(enumerate(iris.target_names)))

#head()will display the first fives rows by deafualt
print(df.head())

#split the dataset into the train data and the test data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)
print("dataset is been splited into train dataset and the test dataset")

#train a decision tree classifier

model =DecisionTreeClassifier()
model.fit(x_train,y_train)

print(x_test)
y_pred = model.predict(x_test)
print(y_pred)

accuracy = accuracy_score(y_test,y_pred)
print(y_test)

print(f'model accuracy:{accuracy *100:.2f}%')

model= LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_train)
print(y_pred)

acc=accuracy_score(y_test, y_pred)
print(y_test)

print(f"model accuracy :{acc*100:.2f} %")
