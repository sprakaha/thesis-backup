import numpy as np
import math
# designed for iterating through array

# Calculate the slope of a line
def slope(x0, y0, x1, y1):
    return (y1 - y0) / (x1 - x0)

# Calculate the angles from the extracted points 
def angleBetweenVectors(x0, y0, x1, y1, x2, y2):
    # (x1, y1) is the central point
    v0 = np.array([(x0 - x1), (y0 - y1)])
    v0N = np.sqrt(np.dot(v0, v0))
    v1 = np.array([(x2 - x1), (y2 - y1)])
    v1N = np.sqrt(np.dot(v1, v1))
    theta = math.acos(np.dot(v0, v1) / (v0N * v1N))

    return theta

# Calculate the 3D angle from the extracted points 
def angleBetweenVectors3D(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    # (x1, y1) is the central point
    v0 = np.array([(x0 - x1), (y0 - y1), (z0 - z1)])
    v0N = np.sqrt(np.dot(v0, v0))
    v1 = np.array([(x2 - x1), (y2 - y1), (z2 - z1)])
    v1N = np.sqrt(np.dot(v1, v1))
    theta = math.acos(np.dot(v0, v1) / (v0N * v1N))

    return theta

# Used for distance between two points in 1 dimension (x0) and (x1)
def distance1d(x0, x1):
    return abs(x0 - x1)

# Used for length/width/displacement between two points (x0, y0) and (x1, y1)
def distance2d(x0, y0, x1, y1):
    return math.sqrt(((x1 - x0)**2 + (y1 - y0)**2))

# Calculate either angular or linear velocity
def velocity(deltaD, deltaT):
    return deltaD / deltaT

# Calculate either angular or linear acceleration
def acceleration(deltaV, deltaT):
    return deltaV / deltaT

# Calculate angle of the vector (x0, y0) - (x1, y1) relative to a horizontal line
def angleReltoPerp(x0, y0, x1, y1):
    # (x1, y1) is the tail of vector
    v0 = np.array([(x0 - x1), (y0 - y1)])
    # create horizontal vector
    h0 = np.array([x1 + 0.2, y1])
    # calculate angle
    v0N = np.sqrt(np.dot(v0, v0))
    h0N = np.sqrt(np.dot(h0, h0))
    theta = math.acos(np.dot(v0, h0) / (v0N * h0N))

    return theta