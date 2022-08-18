import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import tree
import numpy as np
import os



dataframe = pd.read_csv('E:\Visual Code Projects\Python Projects\python\Machine Learning\drug200.csv')
print('\n\nFirst 5 Rows of our Dataframe\n')
print(dataframe.head())

pridictor = dataframe.drop(['Drug'],axis='columns')
target = dataframe.drop(['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K'],axis='columns')
print('\n\nOur predictor attributes\n')
print(pridictor.head())
print('\n\nOur target attribute\n')
print(target.head())
print('\n\nTotal size of Dataframe\n')
print(dataframe.shape)

num_sex = LabelEncoder()
num_bp = LabelEncoder()
num_cholesterol = LabelEncoder()
num_drug = LabelEncoder()
pridictor["Numeric_Sex" ]= num_sex.fit_transform(pridictor["Sex"])
pridictor["Numeric_BP" ]= num_bp.fit_transform(pridictor["BP"])
pridictor["Numeric_Cholesterol" ]= num_cholesterol.fit_transform(pridictor["Cholesterol"])
target["Numeric_Drug"] = num_drug.fit_transform(target['Drug'])
pridictor = pridictor.drop(["Sex","BP","Cholesterol"],axis='columns')
target = target.drop(['Drug'],axis='columns')

print('\n\nNumerical classifiers for our predictor attributes\n')
print(pridictor.head())
print('\n\nNumerical classifiers for our target attributes\n')
print(target.head())

X_train, X_test, y_train, y_test = train_test_split(pridictor, target, train_size =0.75)
model = tree.DecisionTreeClassifier()
model.fit(X_train, y_train)

predict = model.predict(X_test)
length = len(predict)
score = 0
for i in range(length):
    if predict[i] == y_test.Numeric_Drug.values[i]:
        score = score + 1
score = (score/length)*100
print()
print()
print('\n\nHence our model is ' + str(score) + "% efficient\n")
os.system("pause")


