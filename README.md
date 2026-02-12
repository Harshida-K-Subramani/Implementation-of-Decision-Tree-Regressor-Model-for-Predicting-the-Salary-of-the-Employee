# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null values using .isnull() function. 
3. Import LabelEncoder and encode the dataset.
4. Import DecisionTreeRegressor from sklearn and apply the model on the dataset.
5. Predict the values of arrays. 
6. Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset. 
7. Predict the values of array. 8.Apply to new unknown values.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: HARSHIDA K S
RegisterNumber:  212224040108
*/

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
data = pd.read_csv("Salary_EX7.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor,plot_tree
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse = metrics.mean_squared_error(y_test,y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict(pd.DataFrame([[5,6]], columns=x.columns))
plt.figure(figsize=(20, 8))
plot_tree(dt, feature_names=list(x.columns), filled=True)
plt.show()

```

## Output:

## Data head

<img width="270" height="169" alt="exp9(1)" src="https://github.com/user-attachments/assets/eec8a1fd-cb08-4a38-a45b-c07b8e4da672" />

## Data Info

<img width="309" height="166" alt="exp9(2)" src="https://github.com/user-attachments/assets/4c97b18d-6a32-42c7-96bc-ba3a2723b26a" />

## Data Details

<img width="342" height="419" alt="exp9(3)" src="https://github.com/user-attachments/assets/aa90c5cb-81a2-4704-9ef4-f668531fdc0e" />

## Data prediction

<img width="1044" height="385" alt="exp9(4)" src="https://github.com/user-attachments/assets/ec7bae40-d1d5-415a-a52b-463d90f49daf" />


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
