import numpy as np
from helpers import noDivideByZero
# designed for calculating np arrays

def noDivideByZeroArray(x):
    res = []
    for i in x:
        res.append(noDivideByZero(i)) 
    return res 

# Calulate the ratio of two arrays
def ratio(x0, x1):
    x1[np.abs(x1) < 1e-40] = 1e-10
    return x0 / noDivideByZeroArray(x1)

# Calculate the slope of a line
def slope(x0, y0, x1, y1):
    xd = x1 - x0
    xd[np.abs(xd) < 1e-40] = 1e-10
    return y1 - y0 / noDivideByZeroArray(xd)

# Calculate the angles from the extracted points  
def angleBetweenVectors(x0, y0, x1, y1, x2, y2):
    # (x1, y1) is the central point
    v0 = np.array([x0 - x1, y0 - y1]).T
    v1 = np.array([x2 - x1, y2 - y1]).T
    
    # get magnitude of each vector 
    v0N = np.linalg.norm(v0, axis = 1) 
    v1N = np.linalg.norm(v1, axis = 1) 
    
    denom =  np.multiply(v0N, v1N) 
    dotprod = np.sum(v0 * v1, axis = 1)
    theta = np.arccos(dotprod / denom)
    
    return theta

# Used for distance between two points in 1 dimension (x0) and (x1)
def distance1d(x0, x1):
    return abs(x0 - x1)

# Used for length/width/displacement between two points (x0, y0) and (x1, y1)
def distance2d(x0, y0, x1, y1):
    xd = x1 - x0
    yd = y1 - y0
    return np.sqrt(np.square(xd) + np.square(yd))

# Calculate either angular or linear displacement 
def displacement(x1):
    return  x1[1:] - x1[:-1]

# Calculate either angular or linear velocity 
def velocity(x1, T1):
    dx1 = displacement(x1)
    dT1 = displacement(T1)
    return dx1 / noDivideByZeroArray(dT1[:len(dx1)])

# Calculate either angular or linear acceleration  
def acceleration(x1, T1):
    v1 = velocity(x1, T1)
    dv1 = v1[1:] - v1[:-1]
    dT1 = displacement(T1)
    return dv1 / noDivideByZeroArray(dT1[:len(dv1)])

    # Calculate either angular or linear acceleration 
def jerk(x1, T1):
    a1 = acceleration(x1, T1)
    da1 = a1[1:] - a1[:-1]
    dT1 = displacement(T1)
    return da1 / noDivideByZeroArray(dT1[:len(da1)])

# Calculate angle of the vector (x0, y0) - (x1, y1) relative to a horizontal line  
def angleReltoPerp(x0, y0, x1, y1):
    # (x1, y1) is the tail of vector
    v0 = np.array([x0 - x1, y0 - y1]).T
    # create horizontal vector
    h0 = np.array([x1 + 0.2, y1]).T
    # normalize
    v0N = np.linalg.norm(v0, axis = 1) 
    h0N = np.linalg.norm(h0, axis = 1) 
    # calculate angle
    denom =  np.multiply(v0N, h0N) 
    dotprod = np.sum(v0 * h0, axis = 1)
    theta = np.arccos(dotprod / denom)

    return theta

# Calculate inflection point of smoothed differential data 
def getInflectionPoint(x0, y0):
    # compute second derivative
    smooth_d2 = np.gradient(np.gradient(y0))
    # find switching points
    infls = np.where(np.diff(np.sign(smooth_d2)))[0]
    return x0[infls], y0[infls]