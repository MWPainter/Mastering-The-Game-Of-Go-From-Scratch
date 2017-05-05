import random

def selectRandomKey(weightDict):
    """
    For a map from <Type> -> number, return a random key, with probabilities proportional 
    to the weights (the numbers).

    :param weightDict: The dictionary from keys to weights to pick a random key from
    """
    weights = []
    elems = []
    for elem in weightDict:
        weights.append(weightDict[elem])
        elems.append(elem)
    total = sum(weights)
    key = random.uniform(0, total)
    runningTotal = 0.0
    chosenIndex = None
    for i in range(len(weights)):
        weight = weights[i]
        runningTotal += weight
        if runningTotal > key:
            chosenIndex = i
            return elems[chosenIndex]
    raise Exception('Should not reach here')


