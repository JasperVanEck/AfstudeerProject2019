# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 13:54:57 2019

@author: Jasper van Eck
"""

import sys
import numpy as np
import math
import random
import getData
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing 
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import statsmodels.api as sm

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

#Determine the minimum viable X coord to use in the locationFunction
def minXCoord(delay, micDistance):
    ABaccent = SPEEDOFSOUND * delay
    Xb = micDistance / 2
    return math.sqrt(-1*((ABaccent**2 * (ABaccent**2 - 4 * Xb**2))/(4*(4 * Xb**2 - ABaccent**2))))+5

#function to shift coordinates from NAO coords to world/field coords
def coordinateShift(XYnao, XYtarget):
    new = []
    new.append(XYnao[0] + XYtarget[0])
    new.append(XYnao[1] + XYtarget[1])
    return new

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
    return math.radians(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

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

#Create noise to be applied to time delays and add it.
def addGaussianNoise(timeDelays):
    noise = []
    for i in range(len(timeDelays[0])):
        noise.append(np.random.normal(np.mean(timeDelays[:,[i]]), np.std(timeDelays[:,[i]]), len(timeDelays[:,[i]])))
    return timeDelays + np.array(noise).T

#Create noise to be applied to time delays with multiplier.
def gaussianNoise(timeDelays, mult):
    noise = []
    for i in range(len(timeDelays[0])):
        noise.append(np.random.normal(np.mean(timeDelays[:,[i]]), np.std(timeDelays[:,[i]])*mult, len(timeDelays[:,[i]])))
    
    #noise = [[i*mult for i in r] for r in noise]
    return np.array(noise).T

#Create model using sklearn
def trainModel(X, Y):
    #modelSK = AdaBoostClassifier(n_estimators=100, random_state=0)
    modelSK = KNeighborsClassifier(n_neighbors=15, weights='distance')
    #modelSK = LogisticRegression(class_weight = 'balanced')
    modelSK.fit(X, Y)
    
    return modelSK

#Create model using SM
def trainModelSM(X, Y):
    X = sm.add_constant(X)
    #modelSM = sm.OLS(Y, X).fit()
    modelSM = sm.Logit(Y, X).fit()
    
    return modelSM

#Main Function; calls all other functions & stuff
def main(argv):

    #Retrieve Data & Seperate it in usable arrays
    data = getData.getData(25000)
    random.shuffle(data)
    delays = np.array(getData.getTimeDelays(data))
    classification = getData.getClassifications(data)
    #soundSourceLoc = getData.getSoundSourceLocations(data)
    #print(soundSourceLoc)
    naoLocations = getData.getRobotLocations(data)
    
    timeDelays = []
    for i in range(len(delays)):
        timeDelays.append(delayMics(delays[i]))
        
    multiplier = 10
    timeDelays2 = timeDelays + gaussianNoise(np.array(timeDelays), multiplier)

    
    predictedSoundSource = []
    #Determine functions of possible locations    
    for i in range(len(naoLocations)):
        distances = micDistances(naoLocations[i])
        midPoint = midPoints(naoLocations[i])
        #delayMic = delayMics(delays[i])
        delayMic = timeDelays2[i]
        closests = closestsMic(delays[i])
        yPlus1 = []
        yPlus2 = []
        yMin1 = []
        yMin2 = []
        for j in range(len(distances)):
            naoCombos = naoCombo(naoLocations[i], j)
            X_coord = minXCoord(delayMic[j], distances[j])
            X2_coord = X_coord + 20
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

        
        #Arrayify the coordinates of possible sound source locations
        intersections = []
        yPlus1 = np.array(yPlus1)
        yPlus2 = np.array(yPlus2)
        yMin1 = np.array(yMin1)
        yMin2 = np.array(yMin2)
        
        #Create the lines to use for intersection
        lines = []
        for j in range(len(yPlus1)):
            lines.append(line(yPlus1[j], yPlus2[j]))
            lines.append(line(yMin1[j], yMin2[j]))
        
        #Intersect all the lines, except for itself and the minus version
        for j in range(len(lines)):
            l = 0
            if j%2 == 0:
                l = j+2
            else:
                l = j+1
            for k in range(l, len(lines), 1):
                intersections.append(intersection(lines[j], lines[k]))

        goodIntsect = []
        for j in range(len(intersections)):
            if (intersections[j][0] < 12 and intersections[j][0] > -12) and (intersections[j][1] < 12 and intersections[j][1] > -12):
                goodIntsect.append(intersections[j])
        
        #print(goodIntsect)
        #print("----")
        #Average the intersection Positions
        predictedSoundSource.append(averageCoords(goodIntsect))
        
    #print(predictedSoundSource)

    #Append time delays to locations to form complete matrix of data
    #train = np.append(naoLocations, timeDelays2, axis=1)
    train = np.append(naoLocations, predictedSoundSource, axis=1)
    #train = np.append(naoLocations, soundSourceLoc, axis=1)
    #train = predictedSoundSource
    
    #Split data in test and training sets.
    testLength = 1000
    pre_test = train[-testLength:]
    pre_train = train[:len(train)-testLength]
    classTest = classification[-testLength:]
    classTrain = classification[:len(classification)-testLength]

    #Normalize test & training data
    norm_train = preprocessing.normalize(pre_train)
    norm_test = preprocessing.normalize(pre_test)

    #Use calculated sound source location & own location for multiple linear regression
    model = trainModel(norm_train, classTrain)
    #modelSM = trainModelSM(norm_train, classification)

    #Do prediction on test data & print report
    classPred = model.predict(norm_test)
    #classPredProb = model.predict_proba(norm_test)

    #Print the Results
    print(model.__class__.__name__)
    print("The noise multiplier: " + str(multiplier))
    print("The accuracy: " + str(accuracy_score(classTest, classPred)))
    print(classification_report(classTest, classPred, target_names=['Inside','Out of bounds']))
    print("The Mean Squared Error: " + str(mean_squared_error(classTest, classPred)))
    print("Cofusion Matrix: ")
    print("(tn, fp, fn, tp)")
    print(confusion_matrix(classTest, classPred).ravel())

if __name__ == "__main__":
    main(sys.argv)

