# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 01:52:14 2019

@author: Jasper
"""
#SOURCED FROM: https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/barchart.html
#and : https://chrisalbon.com/python/data_visualization/matplotlib_grouped_bar_plot/
import matplotlib.pyplot as plt
import numpy as np


accuracy = (0.505, 0.493, 0.493, 0.504, 0.512, 0.49)
precisionClass0 = (0.27, 0.27, 0.30, 0.30, 0.33, 0.27)
precisionClass1 = (0.73, 0.72, 0.71, 0.70, 0.69, 0.71)
recallClass0 = (0.50, 0.49, 0.53, 0.49, 0.50, 0.48)
recallClass1 = (0.51, 0.49, 0.48, 0.51, 0.52, 0.49)
F1scoreClass0 = (0.35, 0.35, 0.38, 0.37, 0.39, 0.35)
F1scoreClass1 = (0.60, 0.58, 0.57, 0.59, 0.59, 0.58)
meanSquaredError = (0.495, 0.507, 0.507, 0.496, 0.488, 0.51)

ind = np.arange(len(accuracy))  # the x locations for the groups
width = 0.15  # the width of the bars

fig, ax = plt.subplots(figsize=(13,8))
rects1 = ax.bar(2 * ind - 3 * width, accuracy, width, label='Accuracy')
rects2 = ax.bar(2 * ind - 2 * width, precisionClass0, width, label='Precision 0')
rects3 = ax.bar(2 * ind - 1 * width, precisionClass1, width, label='Precision 1')
rects4 = ax.bar(2 * ind - 0 * width, recallClass0, width, label='Recall 0')
rects5 = ax.bar(2 * ind + 1 * width, recallClass1, width, label='Recall 1')
rects6 = ax.bar(2 * ind + 2 * width, F1scoreClass0, width, label='F1-score 0')
rects7 = ax.bar(2 * ind + 3 * width, F1scoreClass1, width, label='F1-score 1')
rects8 = ax.bar(2 * ind + 4 * width, meanSquaredError, width, label='MSE')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Amount of times noise added')
ax.set_ylabel('Scores')
ax.set_title('Score comparison of each noise added')
ax.set_xticks([2,4,6,8,10,12], ['0x', '1x', '2x', '3x', '4x', '5x'])
ax.set_xticklabels((' ', '0x', '1x', '2x', '3x', '4x', '5x', ' '))
ax.legend(ncol=1)

fig.tight_layout()

plt.show()