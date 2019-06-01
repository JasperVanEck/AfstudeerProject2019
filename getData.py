# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 13:54:57 2019

@author: Jasper van Eck
"""

import csv

DATALOCATION = "C:/Users/Jasper/Documents/Schoolwerk/AfstudeerProjectBScAI/2019/Code/UERoboCup-master/TrainingSetGenerator/Saved/ScreenshotMasks/"

#Sourced from master thesis - Repo: https://github.com/S3BASTI3N/robopose
def readFile(dataset_file):
    dataset = []
    for line in open(DATALOCATION + dataset_file):
        string_elements = line[:-1].split(" ")
        float_elements = []
        for x in string_elements:
            float_elements.extend([float(x[2:])])
            
        dataset.extend(float_elements)
    return dataset

def getData(locCount):
    locationData = []
    for i in range(locCount):
        path = "mask%d.txt" % i
        lineData = readFile(path)
        locationData.append(lineData)
	
    return locationData

def getDataDelays():
	with open('delaysData.csv' , 'r') as f:
		reader = csv.reader(f)
		delayData = list(reader)
	
	return delayData

def getTimeDelays(data):
    delays = []
    for i in range(len(data)):
        delays.append(data[i][-3:])
    
    return delays

def getClassifications(data):
    classi = []
    for i in range(len(data)):
        outOfBound = (data[i][18] > 1575 or data[i][18] < 980) or (data[i][19] > 928 or data[i][19] < 30)
        if outOfBound:
            classi.append(1)
        else:
            classi.append(0)
    
    return classi

def getRobotLocations(data):
    locations = []
    for i in range(len(data)):
        loc = data[i][:2]
        loc.extend(data[i][6:8])
        loc.extend(data[i][12:14])
        locations.append(loc)
    
    return locations