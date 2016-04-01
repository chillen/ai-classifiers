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
optimal.build(wine, wines)
naive.build(wine,wines)
linear.build(wine, wines)

print(" -- Heart -- ")
heartTree = dt.build(heart, hearts)
dt.test(heart, heartTree, hearts)
optimal.build(heart, hearts)
naive.build(heart,hearts)
linear.build(heart, hearts)

print(" -- Iris -- ")
irisTree = dt.build(iris, iriss)
dt.test(iris, irisTree, iriss)
optimal.build(iris, iriss)
naive.build(iris,iriss)
linear.build(iris, iriss)
