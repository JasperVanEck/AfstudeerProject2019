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

#Constants
SPEEDOFSOUND = 343#m/s
BALL_COORDS = [1276, 480]

#For 2 NAOs coords, give the distance in m([X,Y],[X,Y])
def naoDistance(nao1, nao2):
    return math.sqrt((nao1[0]+nao2[0])**2+(nao1[1]+nao2[1])**2)

#For a given delay, distance between mics and a given X coordinate, returns the Y coordinate
def locationFunction(X, delay, micDistance):
    ABaccent = SPEEDOFSOUND * delay
    Xb = micDistance / 2
    return math.sqrt(((ABaccent**2)/4)-(Xb**2)+X**2 * ((4*(Xb**2))/Xb**2 - 1))

#function to shift coordinates from NAO coords to world/field coords
def coordinateShift(XYnao, XYtarget):
    return (int(XYnao[0]) + int(XYtarget[0]), XYnao[1] + XYtarget[1])

#Line intersection coordinates, for linear only, sourced from stackOverflow([[],[]],[[],[]])
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
    return x, y

#calculate the angle of the sound source[X,Y]
def angleSoundSource(location):
    return 90 - np.arctan((location[1])/(location[0]))

#Main Function; calls all other functions & stuff
def main(argv):

    #Retrieve Data & Seperate it in usable arrays
    data = getData.getData(10)
    delays = getData.getTimeDelays(data)
    classification = getData.getClassifications(data)
    naoLocations = getData.getRobotLocations(data)
    #print(data)
    print(delays[0])
    print(classification)
    print(naoLocations[0])
    
    """Microphone Distances
    #Part	Location	Name	    X (m)	Y (m)	    Z (m)
    #A	Front	Right	MicroFR	0.0206	-0.0309	0.0986
    #B	Front Left	MicroFL	0.0206	0.0309	0.0986
    #C	Back	Left	MicroRL	-0.0215	0.0558	0.0774
    #D	Back Right	MicroRR	-0.0215	-0.0558	0.0774"""
    #micDisAB = 0.0618 #meter
    #micDisAC = math.sqrt((0.0206+0.0215)**2+(-0.0309-0.0558)**2+(0.0986-0.0774)**2) #0.09868505459
    #micDisAD = math.sqrt((0.0206+0.0215)**2+(-0.0309+0.0558)**2+(0.0986-0.0774)**2)
    #micDisBC = math.sqrt((0.0206+0.0215)**2+(0.0309-0.0558)**2+(0.0986-0.0774)**2)
    #micDisBD = math.sqrt((0.0206+0.0215)**2+(0.0309+0.0558)**2+(0.0986-0.0774)**2)
    #micDisCD = 0.1116 #meter
    #print(micDisAB, " + ", micDisCD)
    
    #X = 1
    #Determine functions of possible locations
    #yPlus = locationFunction(X, delays[0][0], micDisAB)
    #yMin = -1 * yPlus
    
    #Shift to real world coordinates
    #fieldLocation = coordinateShift(locations[0], (X, yPlus))
    
    #Intersect those functions
    #sourceLocation = lineIntersection(fieldLocation, fieldLocation)
    
    #Average the intersects
    #angleSource = angleSoundSource(sourceLocation)
    #print(angleSource)
    
    #Calculate angle & distance from averaged locations
    
    #Use calculated sound source location & own location for multiple linear regression
    #model = regr.trainModel(X, yPlus)
    #modelSM = regr.trainModelSM(X, yPlus)
    
    #model.predict(X)
    #modelSM.predict(X)
    #Check actual sound location vs calculated location, give error etc.
    #print(modelSM.smModelSummary())


if __name__ == "__main__":
    main(sys.argv)

