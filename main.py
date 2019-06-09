# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 13:54:57 2019

@author: Jasper van Eck
"""

import sys
import numpy as np
import math
#import qi
import getData
import regressionModel as regr
from sklearn import preprocessing 

#Constants
SPEEDOFSOUND = 343#m/s

#For 2 NAOs coords, give the distance in m([X,Y],[X,Y])
def naoDistance(nao1, nao2):
    return math.sqrt((nao1[0]+nao2[0])**2+(nao1[1]+nao2[1])**2)

#For a given delay, distance between mics and a given X coordinate, returns the Y coordinate
def locationFunction(X, delay, micDistance):
    ABaccent = SPEEDOFSOUND * delay
    Xb = micDistance / 2
    return math.sqrt((((ABaccent**2)/4)-(Xb**2)) + (X**2 * (((4*(Xb**2))/Xb**2) - 1)))

#function to shift coordinates from NAO coords to world/field coords
def coordinateShift(XYnao, XYtarget):
    new = []
    new.append(XYnao[0] + XYtarget[0])
    new.append(XYnao[1] + XYtarget[1])
    return new

"""Line intersection coordinates, for linear only, sourced from stackOverflow([[],[]],[[],[]])
#https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines-in-python
def lineIntersection(line1, line2):
    xdiff = [line1[0][0] - line1[1][0], line2[0][0] - line2[1][0]]
    ydiff = [line1[0][1] - line1[1][1], line2[0][1] - line2[1][1]]
    
    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y"""

#Line intersection coordinates, for linear only, sourced from stackOverflow([[],[]],[[],[]])
#https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines-in-python
def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return [x,y]
    else:
        return False

#calculate the angle of the sound source[X,Y]
def angleSoundSource(location):
    return 90 - np.arctan((location[1])/(location[0]))

#Calculate the distances between the NAOs
def micDistances(naoLocations):
    distances = [0, 0, 0]
    distances[0] = naoDistance(naoLocations[0:2], naoLocations[2:4])
    distances[1] = naoDistance(naoLocations[0:2], naoLocations[4:])
    distances[2] = naoDistance(naoLocations[2:4], naoLocations[4:])
    return distances

#Determine the mid points of the mics
def midPoints(naoLocations):
    midPoints = []
    midPoints.append([(naoLocations[0] + naoLocations[2])/2, (naoLocations[1] + naoLocations[3])/2])
    midPoints.append([(naoLocations[0] + naoLocations[4])/2, (naoLocations[1] + naoLocations[5])/2])
    midPoints.append([(naoLocations[2] + naoLocations[4])/2, (naoLocations[3] + naoLocations[5])/2])
    return midPoints

#Delays between Mics rather than source and a mic
def delayMics(delays):
    delayMic = []
    delayMic.append(abs(delays[0]-delays[1]))
    delayMic.append(abs(delays[0]-delays[2]))
    delayMic.append(abs(delays[1]-delays[2]))
    return delayMic

#Average the coordinates
def averageCoords(coordsArray):
    x_total = 0
    y_total = 0
    n = len(coordsArray)
    for i in range(n):
        x_total += coordsArray[i][0]
        y_total += coordsArray[i][1]
    x_avgr = x_total / n
    y_avgr = y_total / n
    return [x_avgr, y_avgr]

"""
#create realworld coords from translation matrix and local coords
def localToReal(m, localCoords):
    realCoords = []
    localM = np.matrix([localCoords[0], localCoords[1], 1])
    realM = m * localM.T
    realCoords = realM.T.getA1()[0:2]
    return realCoords
    
#Create a transformation matrix to translate from mic space to real space.([x,y],[x,y],[x,y],[1 or 2])
def translationMatrix(naoCombo, midPoint, closests):
    x = naoDistance(naoCombo[0], midPoint)
    a = 0
    b = 0
    if closests == 1:
        a = np.matrix([[x, 0, -x], [0, 0, 0], [1, 1, 1]])
        b = np.matrix([[naoCombo[0][0], midPoint[0], naoCombo[1][0]], [naoCombo[0][1], midPoint[1], naoCombo[1][1]], [1, 1, 1]])
    else:
        a = np.matrix([[-x, 0, x], [0, 0, 0], [1, 1, 1]])
        b = np.matrix([[naoCombo[0][0], midPoint[0], naoCombo[1][0]], [naoCombo[0][1], midPoint[1], naoCombo[1][1]], [1, 1, 1]]) 
        
    inverse = 0
    try:
        inverse = inv(a)
    except np.linalg.LinAlgError as e:
        print(e)
        print("Fookin error")
    
    print(a)
    #print(b)
    print(inverse)
    print(a*inverse.T)
    #m = np.matmul(b, inverse)
    m = b*inverse
    #print(m)
    return m

#From https://stackoverflow.com/questions/13795682/numpy-error-singular-matrix
def inv(m):
    a, b = m.shape
    if a != b:
        raise ValueError("Only square matrices are invertible.")

    i = np.eye(a, a)
    return np.linalg.lstsq(m, i, rcond=None)[0]
"""

#Determine which mic of pairings is closests
def closestsMic(delays):
    closests = [0, 0, 0]
    if delays[0] > delays[1]:
        closests[0] = 1
    else:
        closests[0] = 2
        
    if delays[0] > delays[2]:
        closests[1] = 1
    else:
        closests[1] = 2
        
    if delays[1] > delays[2]:
        closests[2] = 1
    else:
        closests[2] = 2

    return closests

#create the nao combo 12, 13, 23
def naoCombo(naoLocations, n):
    naoCombo = []
    if n == 0:
        naoCombo = [[naoLocations[0],naoLocations[1]],[naoLocations[2],naoLocations[3]]]
    elif n == 1:
        naoCombo = [[naoLocations[0],naoLocations[1]],[naoLocations[4],naoLocations[5]]]
    else:
        naoCombo = [[naoLocations[2],naoLocations[3]],[naoLocations[4],naoLocations[5]]]
    return naoCombo

#Sourced from: https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

#Sourced from: https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return math.cos(math.radians(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))))

#Rotate a vector by certain degrees counterclockwise
def rotateVector(vector, theta):
    c, s = np.cos(theta), np.sin(theta)
    rotationM = np.array(((c,-s), (s, c)))
    #print(rotationM)
    rotated = rotationM @ vector
    #print(rotated)
    return rotated

#translate & rotate from local to realworld coords
def localToReal(local, naoCombo, midPoint, closests):
    real = []
    angleVec1 = []
    if closests == 1:
        angleVec1 = [naoCombo[0][0]-naoCombo[1][0],naoCombo[0][1]-naoCombo[1][1]]
    else:
        angleVec1 = [naoCombo[1][0]-naoCombo[0][0],naoCombo[1][1]-naoCombo[0][1]]
    
    #print(angleVec1)
    theta = angle_between(angleVec1, [1,0])
    tmp = rotateVector(local, theta)
    #print(tmp)
    real = coordinateShift(tmp,midPoint)
    #print(real)
    return real

#Main Function; calls all other functions & stuff
def main(argv):

    #Retrieve Data & Seperate it in usable arrays
    data = getData.getData(10)
    delays = getData.getTimeDelays(data)
    classification = getData.getClassifications(data)
    soundSourceLoc = getData.getSoundSourceLocations(data)
    print(soundSourceLoc)
    naoLocations = getData.getRobotLocations(data)
    #print(data)
    #print(delays[0])
    #print(classification)
    #print(naoLocations)
        
    predictedSoundSource = []
    X_coord = 10
    X2_coord= 600
    #Determine functions of possible locations    
    for i in range(len(naoLocations)):
        distances = micDistances(naoLocations[i])
        midPoint = midPoints(naoLocations[i])
        delayMic = delayMics(delays[i])
        closests = closestsMic(delays[i])
        yPlus1 = []
        yPlus2 = []
        yMin1 = []
        yMin2 = []
        for j in range(len(distances)):
            naoCombos = naoCombo(naoLocations[i], j)
            
            shift1 = [X_coord]
            coord1 = locationFunction(X_coord, delayMic[j], distances[j])
            shift1.append(coord1)
            yPlus1.append(np.array(localToReal(shift1, naoCombos, midPoint[j], closests[j])))

            shift2 = [X2_coord]
            coord2 = locationFunction(X2_coord, delayMic[j], distances[j])
            shift2.append(coord2)
            yPlus2.append(np.array(localToReal(shift2, naoCombos, midPoint[j], closests[j])))

            #Negative Part
            shift3 = [X_coord]
            shift3.append(-coord1)
            yMin1.append(np.array(localToReal(shift3, naoCombos, midPoint[j], closests[j])))

            shift4 = [X2_coord]
            shift4.append(-coord2)
            yMin2.append(np.array(localToReal(shift4, naoCombos, midPoint[j], closests[j])))

        

        intersectionsPosi = []
        yPlus1 = np.array(yPlus1)
        yPlus2 = np.array(yPlus2)
        yMin1 = np.array(yMin1)
        yMin2 = np.array(yMin2)
        lines = []
        for j in range(len(yPlus1)):
            lines.append(line(yPlus1[j], yPlus2[j]))
            lines.append(line(yMin1[j], yMin2[j]))
            
        for j in range(len(lines)):
            l = 0
            if j%2 == 0:
                l = j+2
            else:
                l = j+1
            for k in range(l, len(lines), 1):
                intersectionsPosi.append(intersection(lines[j], lines[k]))

        
        print(i)
        print(intersectionsPosi)
        print("----")
        
        predictedSoundSource.append(averageCoords(intersectionsPosi))
        
    train = np.append(naoLocations, predictedSoundSource, axis=1)
    norm_train = preprocessing.normalize(train)
    #print(train)
    #print(predictedSoundSource)
    #naoLocations.append(predictedSoundSource)
    #print(naoLocations)
    #Use calculated sound source location & own location for multiple linear regression
    #model = regr.trainModel(X, yPlus)
    #modelSM = regr.trainModelSM(X, yPlus)
    
    #model.predict(X)
    #modelSM.predict(X)
    #Check actual sound location vs calculated location, give error etc.
    #print(modelSM.smModelSummary())


if __name__ == "__main__":
    main(sys.argv)

