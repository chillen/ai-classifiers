import numpy as np

# BEGIN HELPER FUNCTIONS
def headprint(s):
    print(" # "+s+" # \n")

# BEGIN MAIN ALGORITHMS  

# Take in training data and which index the classification is
# Perform a 10-Fold training on it
def Decision_Tree(data, classIndex):
    headprint("Decision Tree")
    trainingGroups = np.array_split(data, 10);
    print(len(trainingGroups[0]))

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
Decision_Tree(wine, 0)