import numpy as np
import decisionTree as dt
import naive
import optimal

# BEGIN HELPER FUNCTIONS
def headprint(s):
    print(" # "+s+" # \n")
   
## I assume this is just a single vector as opposed to matrix (i.e. not diagonal)? Not entirely sure.        
def Linear_Bayesian():
    headprint("Linear Bayesian")
        
## The only difference between OB and NB is the covariance matrix used. In Naive, the 
## covariance matrix is just a diagonal matrix, but in an optimal bayes we use a full
## matrix.        
def Optimal_Bayesian():
    headprint("Optimal Bayesian")
    
wine = np.genfromtxt('data/wine.csv', delimiter=',')
heart = np.genfromtxt('data/heartDisease.csv', delimiter=',')
iris = np.genfromtxt('data/iris.csv', delimiter=',')
wines = ["class", "alcohol", "malic acid", "ash", "alcalinity", "magnesium", "total phenols", "flavanoids", "nonflavanoid phenols", "proanthocyanins", "color", "hue", "OD280", "proline"]
hearts = ["age", "gender", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "dlope", "ca", "thal", "class"]
iriss = ["sepal length", "sepal width", "petal length", "petal width", "class"]

optimal.build(iris, iriss)
