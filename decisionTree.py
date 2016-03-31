import math
from collections import defaultdict
import numpy as np

def Entropy(S): 
    entropy=0
    classes = [sample[0] for sample in S]
    classSum = defaultdict(lambda: 0)
    
    for c in classes:
        classSum[c] += 1
    for c in set(classes):
        p = float(classSum[c]) / len(S)
        entropy -= p*math.log2(p)
    return entropy
 
def split(S, A, min, max):
    data = []
    for sample in S:
        if min <= sample[A] < max:  
            reduce = sample[:A]
            np.append(reduce, sample[A+1:], axis=0)
            data.append(reduce)
    return data
    
def gain(S, A):
    maxV = max([sample[A] for sample in S])
    minV = min([sample[A] for sample in S])
    # binary binning because lazy
    Sv1 = split(S, A, minV, maxV/2)
    Sv2 = split(S, A, maxV/2, maxV+1)
    return Entropy(S) - (( (len(Sv1) / len(S)) * Entropy(Sv1) ) + ( (len(Sv2) / len(S)) * Entropy(Sv2) ))
       

def bestGain(S):
    best = (-1, -9999)
    for i, feature in enumerate(S[0][1:]):
        if gain(S, i+1) > best[1]:
            best = (i+1, gain(S, i+1))
            
    return best[0]    
   
def majorityClass(S):
    classes = defaultdict(lambda:0)
    for c in [sample[0] for sample in S]:
        classes[c] += 1
    maxF = -1
    maxV = -1
    for c in set([sample[0] for sample in S]):
        if maxV < classes[c]:
            maxV = classes[c]
            maxF = c
    return maxF            

def _buildTree(S, headers):
    classes = [sample[0] for sample in S]
    
    if len(S) == 0:
        return "Uncertain"
    
    if len(S[0])==1:
        return majorityClass(S)
        
    best = bestGain(S)
    
    tree = {headers[best]:{}}
    bestHeader = headers[best]
    del (headers[best])
    maxV = max([sample[best] for sample in S])
    minV = min([sample[best] for sample in S])
    Sv1 = split(S, best, minV, maxV/2)
    Sv2 = split(S, best, maxV/2, maxV+1)   
   
    tree[bestHeader]["<= half"] = _buildTree(Sv1, headers[:])
    tree[bestHeader]["> half"] = _buildTree(Sv2, headers[:])
    
    return tree
def buildTree(S, headers): 
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
    return _buildTree(S, headers)
    
def pprint(tree, indent=0):
    if indent == 0:
        key = [k for k in tree][0]
        print(key)