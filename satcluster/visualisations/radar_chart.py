### A Barbara Metzler
## code for radar charts

#import libraries
import geopandas as gp
import numpy as np
import pandas as pd
from math import pi
from sklearn.preprocessing import  QuantileTransformer

from matplotlib.cm import get_cmap
from matplotlib import colors as plt_colors
import matplotlib.pyplot as pltimport geopandas as gp


# input is a dataframe with one column named 'label' which represents cluster labels
# other columns represent external values for demographic and environmental variables

ex_label = ['bui_area', 'count_na', 'avg_bui_na', 'length', 'major_leng', 'min_dist_m',
       'min_dist_t', 'popdense_m', 'nvdi_mean','bui_ori_me', 'label']
ex = ['bui_area', 'count_na', 'avg_bui_na', 'length', 'major_leng', 'min_dist_m',
              'min_dist_t', 'popdense_m', 'nvdi_mean', 'bui_ori_me']

## preprocess data and use QuantileTransformer to tranform features
ns = df[ex_label]
ns[ex] = QuantileTransformer().fit_transform(ns[ex])
n = ns[ex_label].groupby('label').agg('median')
sel = n.copy()


# plot values
SMALL_SIZE = 10
MEDIUM_SIZE = 13
BIGGER_SIZE = 14
​
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


# main plot
fig, axs = plt.subplots(4, 2, figsize=(20, 32), subplot_kw={'projection': 'polar'})

classes = [0,1,2,3,4,5,6,7] #[1,6,5,7,0,3,4,2]
titles = ['Densely populated areas, >36° building orientation', 'Populated areas, <36° building orientation', 'Roads and sparse-moderately populated areas', 'Light vegetation',
        'Buildings surrounded by vegetation', 'Empty land', 'Water', 'Dark dense vegetation']

add_na = [3,5,6,7] ## NA values for clusters with ~0 buildings

cap_labels = ['Building area', 'Building count','Mean building size', 'Length of all roads','Length of major roads', 'Min distance to major roads','Min distance to all roads',
'Population density','Mean NDVI','Mean building orientation']

cap_labels_na = ['Building area', 'Building count','Mean building size', 'Length of all roads','Length of major roads', 'Min distance to major roads','Min distance to all roads',
'Population density','Mean NDVI','Mean building orientation (N.A.)']

cap_labels_na_part1 = ['Building area', 'Building count','Mean building size', 'Length of all roads','Length of major roads']

cap_labels_na_part2 = ['Min distance to major roads','Min distance to all roads',
'Population density','Mean NDVI','Mean building orientation (N.A.)']

# set RGB tuple per class
colors = [(67/255,67/255,67/255), (102/255,102/255,102/255), (204/255,204/255,204/255), (62/255,141/255,129/255), (128/255,128/255,0),
          (138/255,97/255,36/255), (63/255,140/255,203/255), (31/255,117/255,3/255)]


N = len(sel.columns)

angles = [(n / float(N) * 2 * np.pi) +0.5 for n in range(N)]
angles += angles[:1]

for i, ax in zip(classes, axs.flatten()):

    ax.set_xticks(angles[:-1])

    ax.set_ylim(0, 1)
    ax.set_yticks([.25, .5, .75, 1])

    ax.set_yticklabels([.25, .5, .75, 1], color='black', size=10)
    ax.get_yticklabels()[1].set_weight("bold")
    ax.get_yticklabels()[3].set_weight("bold")

    gridlines = ax.yaxis.get_gridlines()
    gridlines[1].set_color("k")
    gridlines[1].set_linewidth(1)

    if i in add_na:
        ax.set_xticklabels(cap_labels_na, color='black', size=13, va="baseline")

    else:
        ax.set_xticklabels(cap_labels, color='black', size=13, va="baseline")


    for label, angle in zip(ax.get_xticklabels(), angles):
        if angle == 0.5:
            label.set_horizontalalignment('left')
        elif angle == 1.1283185307179586:
            label.set_horizontalalignment('left')
        elif angle == 1.7566370614359172:
            label.set_horizontalalignment('center')
        elif angle == 2.384955592153876:
            label.set_horizontalalignment('right')
        elif angle ==  3.0132741228718345:
            label.set_horizontalalignment('right')
        elif angle == 3.641592653589793:
            label.set_horizontalalignment('right')
        elif angle == 4.269911184307752:
            label.set_horizontalalignment('right')
        elif angle == 5.526548245743669:
            label.set_horizontalalignment('left')
        elif angle == 6.154866776461628:
            label.set_horizontalalignment('left')


    ax.set_rlabel_position(10)

    ax.set_title(titles[i], size=16, weight='bold', pad=20)

    values = sel.loc[i].values.flatten().tolist()
    values += values[:1]

    ax.bar(angles, values, width=0.4, color=colors[i])

fig.set_facecolor('white')
fig.subplots_adjust(hspace=0.35)
fig.subplots_adjust(wspace=0.35)

plt.show()
