import random

def selectRandomKey(weightDict, default_action):
    """
    For a map from <Type> -> number, return a random key, with probabilities proportional 
    to the weights (the numbers).

    Args:
        weightDict: The dictionary from keys to weights to pick a random key from
        default_action: Default action (for if only 0's are passed in)
    """
    weights = []
    elems = []
    for elem in weightDict:
        weights.append(weightDict[elem])
        elems.append(elem)
    total = sum(weights)
    if total == 0.0:
        return default_action
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


