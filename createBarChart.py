# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 01:52:14 2019

@author: Jasper
"""
#SOURCED FROM: https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/barchart.html
#and : https://chrisalbon.com/python/data_visualization/matplotlib_grouped_bar_plot/
import matplotlib.pyplot as plt
import numpy as np

"""
         & Accuracy & Precision & Recall & F1-score & MSE & N \\ \hline
        No noise & 0.59 & 0.35 & 0.49 & 0.41 & 0.31 & 35 \\ \hline
        1x Stdr. dev. & 0.50 & 0.33 & 0.43 & 0.38 & 0.50 & 35 \\ \hline
        2x Stdr. dev. & 0.54 & 0.37 & 0.43 & 0.39 & 0.46 & 35 \\ \hline
        3x Stdr. dev. & 0.48 & 0.33 & 0.46 & 0.38  & 0.52 & 35 \\ \hline
        4x Stdr. dev. & 0.55 & 0.38 & 0.46 & 0.42 & 0.45 & 35 \\ \hline
        5x Stdr. dev. & 0.51 & 0.35 & 0.49 & 0.41 & 0.49 & 35 \\ \hline
        
         & Accuracy & Precision & Recall & F1-score & MSE & N \\ \hline
        No noise & 0.59 & 0.68 & 0.71 & 0.69 & 0.31 & 65 \\ \hline
        1x Stdr. dev. & 0.50 & 0.64 & 0.54 & 0.58 & 0.50 & 65 \\ \hline
        2x Stdr. dev. & 0.54 & 0.66 & 0.60 & 0.63 & 0.46 & 65 \\ \hline
        3x Stdr. dev. & 0.48 & 0.63 & 0.49 & 0.55 & 0.52 & 65 \\ \hline
        4x Stdr. dev. & 0.55 & 0.67 & 0.60 & 0.63 & 0.45 & 65 \\ \hline
        5x Stdr. dev. & 0.51 & 0.65 & 0.52 & 0.58 & 0.49 & 65 \\ \hline

"""
accuracy = (0.59, 0.50, 0.54, 0.48, 0.55, 0.51)
precisionClass0 = (0.35, 0.33, 0.37, 0.33, 0.38, 0.35)
precisionClass1 = (0.68, 0.64, 0.66, 0.63, 0.67, 0.65)
recallClass0 = (0.49, 0.43, 0.43, 0.46, 0.46, 0.49)
recallClass1 = (0.71, 0.54, 0.60, 0.49, 0.60, 0.52)
F1scoreClass0 = (0.41, 0.38, 0.39, 0.38, 0.42, 0.41)
F1scoreClass1 = (0.69, 0.58, 0.63, 0.55, 0.63, 0.58)
meanSquaredError = (0.31, 0.50, 0.46, 0.52, 0.45, 0.49)

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