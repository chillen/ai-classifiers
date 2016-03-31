import numpy as np
from collections import defaultdict

def _build(S, headers):
    splitArrays = np.array_split(S, 2)
    training = splitArrays[0]
    testing = splitArrays[1]
    
    T = defaultdict(lambda: [])
    for sample in training:
        c = sample[0]
        T[c].append(sample)

    sums = {}
    M = {}
    k = 0
    
    for key in T:
        k = len(T[key][0]) - 1 # store the length for later
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
        E[key] = (1/(len(T[key])-1)) * sums[key]
        
    # TODO Diagonal the matrix
                

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