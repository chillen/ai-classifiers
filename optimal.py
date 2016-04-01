import numpy as np
from collections import defaultdict
import math
import random

def mahaDistance(x, M, E):
    t1 = np.nan_to_num(np.transpose(x-M))
    try:
        t2 = E.getI()
    except:
        t2 = E
    t3 = x-M
    t4 =t1.dot(t2)
    return t4.dot(t3)

def getClassValue(Ea, Eb, Ma, Mb, x):
    Ea = np.matrix(Ea)
    Eb = np.matrix(Eb)
    Ma = np.matrix(Ma).T
    Mb = np.matrix(Mb).T
    x = np.matrix(x).T
    np.seterr(invalid='ignore', divide='ignore')
    return (np.log(np.linalg.det(Eb)) - np.log(np.linalg.det(Ea)) + mahaDistance(x, Mb, Eb) - mahaDistance(x, Ma, Ea))[0]

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
    return {"E": E, "M": M}

# Not sure how pairwise works. I'm going to just pull one class at random and one I know is correct
def testClassifier(training, testing):
    classifier = getClassifier(training)
    totalTest = 0
    successes = 0
    for test in testing:
        for sample in test:
            actualClass = sample[0]
            possibleClasses = set([classes for classes in classifier["E"]])
            classA = possibleClasses.pop()
            while len(possibleClasses) > 0:
                classB = possibleClasses.pop()
                Ea = classifier["E"][classA]
                Ma = classifier["M"][classA]
                Eb = classifier["E"][classB]
                Mb = classifier["M"][classB]

                val = getClassValue(Ea, Eb, Ma, Mb, sample[1:])
                if val <= 0:
                    classA = classB
            if classA == actualClass:
                successes +=1
            totalTest += 1
    return successes/totalTest

def _build(S, headers, leaveoneout, k=10):
    if leaveoneout:
        k = len(headers)-1
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


def build(S, headers, leaveoneout=False):
     # Find the class header
    print("# Begin building classifiers: Optimal")
    classIndex = -1
    for i, h in enumerate(headers):
        if h == "class":
            classIndex = i

    # Swap the position 0 with the class
    if classIndex != 0:
        S[:,[0,classIndex]] = S[:,[classIndex,0]]
        headers[0], headers[classIndex] = headers[classIndex], headers[0]
    np.random.shuffle(S)
    _build(S, headers,leaveoneout)
