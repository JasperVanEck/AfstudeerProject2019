# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 13:54:57 2019

@author: Jasper van Eck
"""

import csv

DATALOCATION = "C:/Users/Jasper/Documents/Schoolwerk/AfstudeerProjectBScAI/2019/Code/Locations/"

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

def getDataLocations(locCount):
    locationData = []
    for i in range(locCount):
        path = "location%d.txt" % i
        lineData = readFile(path)
        locationData.append(lineData)
	
    return locationData

def getDataDelays():
	with open('delaysData.csv' , 'r') as f:
		reader = csv.reader(f)
		delayData = list(reader)
	
	return delayData
