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
BALL_COORDS = [12.76, 4.80]

#For 2 NAOs coords, give the distance in m([X,Y],[X,Y])
def naoDistance(nao1, nao2):
    return math.sqrt((nao1[0]+nao2[0])**2+(nao1[1]+nao2[1])**2)

#For a given delay, distance between mics and a given X coordinate, returns the Y coordinate
def locationFunction(X, delay, micDistance):
    ABaccent = SPEEDOFSOUND * delay
    Xb = micDistance / 200
    return math.sqrt(((ABaccent**2)/4)-(Xb**2)+X**2 * ((4*(Xb**2))/Xb**2 - 1))

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

#Main Function; calls all other functions & stuff
def main(argv):

    #Retrieve Data & Seperate it in usable arrays
    data = getData.getData(10)
    delays = getData.getTimeDelays(data)
    classification = getData.getClassifications(data)
    naoLocations = getData.getRobotLocations(data)
    #print(data)
    #print(delays[0])
    #print(classification)
    #print(naoLocations[0])
    
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
    
    X_coord = 10
    X2_coord= 15
    #Determine functions of possible locations    
    for i in range(len(naoLocations)):
        distances = micDistances(naoLocations[i])
        midPoint = midPoints(naoLocations[i])
        delayMic = delayMics(delays[i])
        yPlus1 = []
        yPlus2 = []
        yMin1 = []
        yMin2 = []
        for j in range(len(distances)):
            shift1 = [X_coord] 
            shift1.append(locationFunction(X_coord, delayMic[j], distances[j]))
            yPlus1.append(coordinateShift(shift1, BALL_COORDS))
            shift2 = [X2_coord]
            shift2.append(locationFunction(X2_coord, delayMic[j], distances[j]))
            yPlus2.append(coordinateShift(shift2, BALL_COORDS))
        
        for j in range(len(yPlus1)):
            shift1 = [-yPlus1[j][0]]
            shift1.append(-yPlus1[j][1])
            yMin1.append(shift1)
            shift2 = [-yPlus2[j][0]]
            shift2.append(-yPlus2[j][1])
            yMin2.append(shift2)
        
        #print(yPlus1)
        intersectionsPosi = []
        intersectionsNegi = []
        line1Pos = line(yPlus1[0], yPlus2[0])
        line2Pos = line(yPlus1[1], yPlus2[1])
        line3Pos = line(yPlus1[2], yPlus2[2])
        intersectionsPosi.append(intersection(line1Pos, line2Pos))
        intersectionsPosi.append(intersection(line1Pos, line3Pos))
        intersectionsPosi.append(intersection(line2Pos, line3Pos))
        print(intersectionsPosi)
        line1Neg = line(yMin1[0], yMin2[0])
        line2Neg = line(yMin1[1], yMin2[1])
        line3Neg = line(yMin1[2], yMin2[2])
        intersectionsNegi.append(intersection(line1Neg, line2Neg))
        intersectionsNegi.append(intersection(line1Neg, line3Neg))
        intersectionsNegi.append(intersection(line2Neg, line3Neg))
        print(intersectionsNegi)
        
    
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

