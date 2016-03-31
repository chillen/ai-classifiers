import numpy as np
import math
from collections import defaultdict


# Takes dataset S and index of class i
def Entropy(S, index):
    entropy = 0
    classes = defaultdict(lambda: 0)
    
    for row in S:
        classes[row[index]] += 1
                
    for c in classes:
        prob = c / len(S)
        if prob != 0:
            entropy -= prob * math.log(prob, 2)
    return entropy
        
# Returns a dataset where everything under max is returned with that attribute removed
def Split(S, index, min, max):
    data = []
    for sample in S:
        if min <= sample[index] < max:
            reduced = sample[:index]
            reduced = np.append(reduced, sample[index+1:])
            data.append(reduced)
    return data  
     
def Gain(S, A, index): 
    maxV = -99999
    minV = 99999
    for row in S:
        maxV = max(maxV, row[A])
        minV = min(minV, row[A])
    Sv1 = Split(S, A, minV, maxV*.3)
    Sv2 = Split(S, A, maxV*.3, maxV*.6)
    Sv3 = Split(S, A, maxV*.6, maxV+1) # Go a bit past the max here
    return Entropy(S, index) - math.fsum([Entropy(Sv1, index), Entropy(Sv2, index), Entropy(Sv3, index)])
    
# Returns an unformatted decision tree; index is the index of the classification    
def Build_Tree(S, Node, index, wines): 
    bestGain = (-1,-9999)
    
    for i, feature in enumerate(S[0]):
        if i != index:
            if Gain(S, i, index) > bestGain[1]:
                bestGain = (i, bestGain[1])
        
    if 0 < bestGain[0] < len(wines): 
        bestGain = (bestGain[0], bestGain[1], wines[bestGain[0]])
        print(wines[bestGain[0]])
        wines.remove(wines[bestGain[0]])          
    
    # Find min and max
    maxV = -99999
    minV = 99999
    
    for row in S:
        maxV = max(maxV, row[bestGain[0]])
        minV = min(minV, row[bestGain[0]])
    
    if bestGain[0] < index:
        index -= 1
        
    if len(S[0]) == 2:
        return [bestGain[2]]
    
    Sv1 = Split(S, bestGain[0], minV, maxV*.3)
    Sv2 = Split(S, bestGain[0], maxV*.3, maxV*.6)
    Sv3 = Split(S, bestGain[0], maxV*.6, maxV+1) # Go a bit past the max here
        
    if len(Sv1) > 0 and (Entropy(Sv1, index) == 1 or Entropy(Sv1, index) == 0):
        return [bestGain[2]]
    if len(Sv2) > 0 and (Entropy(Sv2, index) == 1 or Entropy(Sv2, index) == 0):
        return [bestGain[2]]
    if len(Sv3) > 0 and (Entropy(Sv2, index) == 1 or Entropy(Sv2, index) == 0):
        return [bestGain[2]]  
          
    if len(Sv1) > 0:    
        Node.append(Build_Tree(Sv1, [], index, wines))
    if len(Sv2) > 0:    
        Node.append(Build_Tree(Sv2, [], index, wines))
    if len(Sv3) > 0:    
        Node.append(Build_Tree(Sv3, [], index, wines))
    
    # Add a child for every possibility

    return Node    

def prettyPrint(tree, depth):
    if isinstance(tree, list):
        for node in tree:
              prettyPrint(node, depth+1)
    else:
        print('-'*depth, tree)