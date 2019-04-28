# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 13:54:57 2019

@author: Jasper van Eck
"""

import csv

def getDataLocations():
	with open('locationsData.csv', 'r') as f:
		reader = csv.reader(f)
		locationData = list(reader)
	
	return locationData

def getDataDelays():
	with open('delaysData.csv' , 'r') as f:
		reader = csv.reader(f)
		delayData = list(reader)
	
	return delayData
