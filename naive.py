import numpy as np
from collections import defaultdict
import math
import random

def mahaDistance(x, M, E):
    t1 = np.nan_to_num(np.transpose(x-M))
    try:
        t2 = np.nan_to_num(E.getI())
    except:
        t2 = np.nan_to_num(E)
    t3 = np.nan_to_num(x-M)
    t4 = np.nan_to_num(t1.dot(t2))
    return np.nan_to_num(t4.dot(t3))

def getClassValue(Ea, Eb, Ma, Mb, x):
    Ea = np.matrix(Ea)
    Eb = np.matrix(Eb)
    Ma = np.matrix(Ma).T
    Mb = np.matrix(Mb).T
    x = np.matrix(x).T
    return np.nan_to_num(np.log(Eb)) - np.nan_to_num(np.log(Ea)) + mahaDistance(x, Mb, Eb) - mahaDistance(x, Ma, Ea)

def getClassifier(training):
    T = defaultdict(lambda: [])
    for sample in training:
        c = sample[0]
        T[c].append(sample)

    sums = {}
    M = {}
    
    for key in T:
        sums[key] = []
        for sample in T[key]:
            for i, feature in enumerate(sample[1:]):
                if len(sums[key]) > i:
                    sums[key][i] += feature
                else:
                    sums[key].append(feature)
    for key in sums:
        M[key] = []
        for element in sums[key]:
           M[key].append(element/len(T[key])) 
    
    # Mean vectors calculated, find the sum of (xi - Ma)(xi-Ma)^T
    sums = {}
    for key in T:
        summable = []
        for x in [sample[1:] for sample in T[key]]:
            x1 = np.subtract(x, M[key])
            x2 = np.transpose(x1)
            summable.append( np.outer(x1,x2) )
        sums[key] = np.sum(summable, axis=0)
        
    E = {}
    
    for key in T:
        if len(T[key]) != 1:
            E[key] = (1/(len(T[key])-1)) * sums[key]
    for key in T:
        for r, row in enumerate(E[key]):
            for c, col in enumerate(row):
                if r != c:
                    E[key][r][c] = 0
    return {"E": E, "M": M}

# Not sure how pairwise works. I'm going to just pull one class at random and one I know is correct
def testClassifier(training, testing):
    classifier = getClassifier(training)
    totalTest = 0
    successes = 0
    for test in testing:
        for sample in test:
            actualClass = sample[0]
            Ea = classifier["E"][actualClass]
            Ma = classifier["M"][actualClass]
        
            # Choose a random other class
            possible = random.sample(list(classifier["E"]),2)
            if possible[0] == actualClass:
                otherClass = possible[1]
            else:
                otherClass = possible[0]
            
            Eb = classifier["E"][otherClass]
            Mb = classifier["M"][otherClass]
            
            val = getClassValue(Ea, Eb, Ma, Mb, sample[1:])
            
            if val.any() > 0:
                successes += 1
            totalTest += 1
    return successes/totalTest           
  
def _build(S, headers, k=10):
    splitArrays = np.array_split(S, k)
    bestTestScore = -99999
    bestTestIndex = -1
    testing = []
    for i in range(0, k):
        for j in range(0, k):
            if j == i:
                training = splitArrays[j]
            else:
                testing.append(splitArrays[j])
        testScore = testClassifier(training, testing)
        print("Completed a test with accuracy of: " + str(testScore))
        if testScore > bestTestScore:
            bestTestScore = testScore
            bestTestIndex = i
            
    print("Best Accuracy Found: " + str(bestTestScore))
                

def build(S, headers):
     # Find the class header
    classIndex = -1
    for i, h in enumerate(headers):
        if h == "class":
            classIndex = i
            
    # Swap the position 0 with the class
    if classIndex != 0:
        S[:,[0,classIndex]] = S[:,[classIndex,0]]
        headers[0], headers[classIndex] = headers[classIndex], headers[0]
    np.random.shuffle(S)
    _build(S, headers)