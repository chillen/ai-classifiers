import numpy as np
import decisionTree as dt

# BEGIN HELPER FUNCTIONS
def headprint(s):
    print(" # "+s+" # \n")

## 1. Partition the training data into two sets, belonging to class A and class B
## 2. Calculate normal distribution for each set
## 3. Form the classification equation
def Naive_Bayesian():
    headprint("Naive Bayesian")
   
## I assume this is just a single vector as opposed to matrix (i.e. not diagonal)? Not entirely sure.        
def Linear_Bayesian():
    headprint("Linear Bayesian")
        
## The only difference between OB and NB is the covariance matrix used. In Naive, the 
## covariance matrix is just a diagonal matrix, but in an optimal bayes we use a full
## matrix.        
def Optimal_Bayesian():
    headprint("Optimal Bayesian")
    
wine = np.genfromtxt('data/wine.csv', delimiter=',')
wines = ["class", "alcohol", "malic acid", "ash", "alcalinity", "magnesium", "total phenols", "flavanoids", "nonflavanoid phenols", "proanthocyanins", "color", "hue", "OD280", "proline"]
dt.prettyPrint(dt.Build_Tree(wine, [], 0, wines), 1)