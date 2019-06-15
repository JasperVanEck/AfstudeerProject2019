# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 01:52:14 2019

@author: Jasper
"""
#SOURCED FROM: https://matplotlib.org/3.1.0/gallery/lines_bars_and_markers/barchart.html
#and : https://chrisalbon.com/python/data_visualization/matplotlib_grouped_bar_plot/
import matplotlib.pyplot as plt
import numpy as np


men_means = (20, 35, 30, 35, 27, 12)
women_means = (25, 32, 34, 20, 25, 10)

ind = np.arange(len(men_means))  # the x locations for the groups
print(ind)
width = 0.15  # the width of the bars

fig, ax = plt.subplots(figsize=(13,7))
rects1 = ax.bar(2 * ind - 3 * width, men_means, width, label='Accuracy')
rects2 = ax.bar(2 * ind - 2 * width, women_means, width, label='Precision Class 0')
rects3 = ax.bar(2 * ind - 1 * width, women_means, width, label='Precision Class 1')
rects4 = ax.bar(2 * ind - 0 * width, women_means, width, label='Recall Class 0')
rects5 = ax.bar(2 * ind + 1 * width, women_means, width, label='Recall Class 1')
rects6 = ax.bar(2 * ind + 2 * width, women_means, width, label='F1-score Class 0')
rects7 = ax.bar(2 * ind + 3 * width, women_means, width, label='F1-score Class 1')
rects8 = ax.bar(2 * ind + 4 * width, women_means, width, label='MSE')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Amount of times noise added')
ax.set_ylabel('Scores')
ax.set_title('Score comparison of each noise added model')
ax.set_xticks([2,4,6,8,10,12], ['0x', '1x', '2x', '3x', '4x', '5x'])
ax.set_xticklabels((' ', '0x', '1x', '2x', '3x', '4x', '5x'))
ax.legend()

fig.tight_layout()

plt.show()