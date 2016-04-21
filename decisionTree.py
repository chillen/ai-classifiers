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
    safeHeaders = headers[:]
    del (safeHeaders[best])
    maxV = max([sample[best] for sample in S])
    minV = min([sample[best] for sample in S])
    half = minV + (maxV-minV) / 2
    Sv1 = split(S, best, minV, half)
    Sv2 = split(S, best, half, maxV+1)

    tree[bestHeader]["<= half"] = _buildTree(Sv1, safeHeaders[:])
    tree[bestHeader]["> half"] = _buildTree(Sv2, safeHeaders[:])

    return tree

def build(S, headers):
    # Find the class header
    print(" # Building Tree")
    classIndex = -1
    for i, h in enumerate(headers):
        if h == "class":
            classIndex = i

    # Swap the position 0 with the class
    if classIndex != 0:
        S[:,[0,classIndex]] = S[:,[classIndex,0]]
        headers[0], headers[classIndex] = headers[classIndex], headers[0]
    np.random.shuffle(S)
    print("# Finished Building Tree")
    return _buildTree(S, headers)

# Returns either a class index or uncertainty
def traverse(S, sample, tree, headers):
    if not isinstance(tree, dict):
        return tree
    head = [k for k in tree][0]

    headIndex = headers.index(head)
    # Find max and min to use for calculating which side of the tree to traverse
    maxV = max([sample[headIndex] for sample in S])
    minV = min([sample[headIndex] for sample in S])
    half = minV + (maxV-minV) / 2

    if sample[headIndex] <=  half:
        return traverse(S, sample, tree[head]["<= half"], headers)
    else:
        return traverse(S, sample, tree[head]["> half"], headers)

def test(S, tree, headers):
    print("# Begining DT Tests")
    totalTests = 0
    successes = 0
    for sample in S:
        classIndex = headers.index("class")
        actualClass = sample[classIndex]
        resultClass = traverse(S, sample, tree, headers)

        #print("Test: ", totalTests)
        #print("  -- Actual Class: ", actualClass)
        #print("  -- Result Class: ", resultClass)
        if actualClass == resultClass:
            successes += 1
        totalTests += 1
    print("Successes: ", successes)
    print("Total: ", totalTests)
    print("Success Rate: %0.2f" % (successes/totalTests))
    print("# DT Tests Complete")
    return (successes/totalTests, successes, totalTests)

def pprint(tree, indent=0):
    if indent == 0:
        key = [k for k in tree][0]
        print(key)
