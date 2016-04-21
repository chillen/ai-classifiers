import numpy as np
import decisionTree as dt
import naive
import optimal
import linear

wine = np.genfromtxt('data/wine.csv', delimiter=',')
heart = np.genfromtxt('data/heartDisease.csv', delimiter=',')
iris = np.genfromtxt('data/iris.csv', delimiter=',')
wines = ["class", "alcohol", "malic acid", "ash", "alcalinity", "magnesium", "total phenols", "flavanoids", "nonflavanoid phenols", "proanthocyanins", "color", "hue", "OD280", "proline"]
hearts = ["age", "gender", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "dlope", "ca", "thal", "class"]
iriss = ["sepal length", "sepal width", "petal length", "petal width", "class"]

print(" -- Wines -- ")
wineTree = dt.build(wine, wines)
dt.test(wine, wineTree, wines)
optimal.build(wine, wines, leaveoneout=False)
naive.build(wine,wines, leaveoneout=False)
linear.build(wine, wines, leaveoneout=False)

print(" -- Heart -- ")
heartTree = dt.build(heart, hearts)
dt.test(heart, heartTree, hearts)
optimal.build(heart, hearts, leaveoneout=False)
naive.build(heart,hearts, leaveoneout=False)
linear.build(heart, hearts, leaveoneout=False)

print(" -- Iris -- ")
irisTree = dt.build(iris, iriss)
dt.test(iris, irisTree, iriss)
optimal.build(iris, iriss, leaveoneout=False)
naive.build(iris,iriss, leaveoneout=False)
linear.build(iris, iriss, leaveoneout=False)

# print(wineTree)
# print(heartTree)
# print(irisTree)
