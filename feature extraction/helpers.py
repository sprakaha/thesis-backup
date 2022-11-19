def selectIfGreater(x , y, val):
    nx = []
    ny = []
    for i in range(len(x)):
        if y[i] > val:
            nx.append(x[i])
            ny.append(y[i])
    return nx, ny

def selectIfLess(x , y, val):
    nx = []
    ny = []
    for i in range(len(x)):
        if y[i] < val:
            nx.append(x[i])
            ny.append(y[i])
    return nx, ny

def getDistance(x):
    if len(x) > 1: 
        dist = 0
        for i in range(len(x) - 1):
            dist += x[i + 1] - x[i]
        return dist
    else: 
        return 0

def noDivideByZero(x):
    if x == 0.0:
        return 1e-10
    else: 
        return x
