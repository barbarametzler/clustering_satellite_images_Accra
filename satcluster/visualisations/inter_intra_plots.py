## A Barbara Metzler
## code for inter and intra cluster distance plots

# import packages
import pandas as pd
import numpy as np
import geopandas as gp

import seaborn as sns

from math import sqrt
from decimal import Decimal
from statistics import mean
import numpy as np
import random
import itertools
from itertools import zip_longest
from itertools import combinations
from functools import reduce
from itertools import groupby
import scipy
import itertools

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# if opened in a Jupyter notebook
#%matplotlib inline



########################################################

## helper functions


def euclid_dist(x2,x1,y2,y1):
    """returns the distance between points (x1,y1) and (x2,y2)"""
    return sqrt(((x2-x1)**2) + ((y2-y1)**2))

def e_d(arra, arrb):
    dist = scipy.spatial.distance.euclidean(arra, arrb)
    return dist

def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)

def rCombos(arr, r):
    return list(combinations(arr, r))


def within_cluster_var(clusters, centroids):
    """takes a list of tuples of (x,y) pairs and cluster assignments, and a list of centroids, and computes WCV"""
    #within cluster variation = sum of squared deviations from cluster mean
    squared_dev = [] #list to store within cluster sum of squared deviations
    for item in clusters:
        squared_dev.append((euclid_dist(item[0][0], centroids[item[1]][0], item[0][1], centroids[item[1]][1])))
    sumsq=0
    for dev in squared_dev:
        sumsq += dev*dev
    #wcv = within cluster variation
    wcv = sumsq
    print('WCV: ' + str(wcv))
    return wcv, squared_dev

def get_centroids(features, label_name):
    features['label'] = lab[label_name]
    ep5 = features.groupby('label').mean()
    centroids = ep5.to_records(index=False)
    cen = list(centroids)
    return cen

def get_points(features, label_name):
    features['label'] = lab[label_name]
    points = features.iloc[:,0:2].to_records(index=False)
    poi = list(zip(points, features['label'])) # pcs8['label']))
    return poi

def between_cluster_var(centroids):
    """takes the coordinates of k centroids as a list of tuples, and computes between cluster variation"""
    #get all combinations of centroid pairs get compute sum squared distance between them
    centroid_combos = rCombos(centroids,2) #every combination of pairs of 2
    bc_dist = [] #list to store between cluster distances
    for combo in centroid_combos:
        bc_dist.append(euclid_dist(combo[1][0],combo[0][0],combo[1][1],combo[0][1]))
    bcv = sum(bc_dist) #sum of distance between every combination of centroids
    print('BCV: ' + str(bcv))
    return bcv, bc_dist, centroid_combos



########################################################

# Data: df with PC reduced features
d = pd.read_csv('../exp/features/k8_imgnet_lr00001_epoch_20_pc256_l2.csv', header=None)

# Labels: df with labels of clusters
labels = pd.read_csv('../all_external_data_labels.csv')

# add label (make sure they have the same index)
d['label'] = labels['k8_label'].values



#########################################################
## scatter plot of PC1 and PC2

colors = [(67/255,67/255,67/255), (102/255,102/255,102/255), (204/255,204/255,204/255), (62/255,141/255,129/255), (128/255,128/255,0),
          (138/255,97/255,36/255), (63/255,140/255,203/255), (31/255,117/255,3/255)]

f, ax = plt.subplots(1, figsize=(15, 10))

one = plt.scatter(d.loc[d['label'] == 0][0], d.loc[d['label'] == 0][1], c=colors[0], s=0.5)#, ax=ax)
two = plt.scatter(d.loc[d['label'] == 1][0], d.loc[d['label'] == 1][1], c=colors[1], s=0.5) #,ax=ax)
three = plt.scatter(d.loc[d['label'] == 2][0], d.loc[d['label'] == 2][1], c=colors[2], s=0.5) #, ax=ax)
four = plt.scatter(d.loc[d['label'] == 3][0], d.loc[d['label'] == 3][1], c=colors[3], s=0.5) #,ax=ax)
five = plt.scatter(d.loc[d['label'] == 4][0], d.loc[d['label'] == 4][1], c=colors[4], s=0.5) #,ax=ax)
six = plt.scatter(d.loc[d['label'] == 5][0], d.loc[d['label'] == 5][1], c=colors[5], s=0.5) #, ax=ax)
seven = plt.scatter(d.loc[d['label'] == 6][0], d.loc[d['label'] == 6][1], c=colors[6], s=0.5) #, ax=ax)
eight = plt.scatter(d.loc[d['label'] == 7][0], d.loc[d['label'] == 7][1], c=colors[7], s=0.5) #, ax=ax)


plt.legend((one, two, three, four, five, six, seven, eight),
       (['Densely populated areas, >36° building orientation', 'Densely populated areas, <36° building orientation', 'Roads and sparse-moderately populated areas', 'Light vegetation',
    'Buildings surrounded by vegetation', 'Empty land', 'Water', 'Dark dense vegetation'] ), markerscale=15,
       fontsize=15,  frameon=False) # markerscale=15,
plt.ylabel('PC2')
plt.xlabel('PC1')
#plt.title('Image features plotted on PC1 and PC2')
plt.axis('off')
plt.show()


#########################################################
## inter and intra cluster distance heatmaps

## calculate centroids per cluster
means = d.groupby('label').mean()
centroids = means.to_records(index=False)
cen = list(centroids)

bcv, bc_dist, centroid_combos = between_cluster_var(cen)

## create DataFrame for plot

# inter distances
dist = pd.DataFrame(bc_dist, columns=['distances'])
dist['combos'] = list(rCombos([0,1,2,3,4,5,6,7], 2))
dist.describe()
d_ = pd.DataFrame()
d_['distances'] = [0,0,0,0,0,0,0,0]
d_['combos'] =[(0,0), (1,1), (2,2), (3,3), (4,4), (5,5), (6,6), (7,7)]
dist = dist.append(d_)
dist['combos'] = dist['combos'].astype(str)
dist['cl1'] = dist['combos'].str.split(",", expand=True)[0]
dist['cl2'] = dist['combos'].str.split(",", expand=True)[1]
dist['cl1'] = dist['cl1'].str.replace("(", "")
dist['cl2'] = dist['cl2'].str.replace(")", "")


# intra distances
# get PC values (remove label column)
points = d.iloc[:,0:-1].to_records(index=False)
poi = list(zip(points, d['label']))

wcv, sdev = within_cluster_var(poi, cen)
d['sdev'] = sdev
intra = pd.DataFrame(d.groupby('label')['sdev'].sum())

## normalise by counts per cluster
count = pd.DataFrame(d['label'].value_counts())
count['index'] = count.index
count = count.sort_index()
intra['count'] = count['label']
intra['sdev_norm'] = intra['sdev']/intra['count']


### inter and intra-distance plot

plt.gcf().subplots_adjust(bottom=0.5)

x_axis_labels = ['Densely populated areas, >36° building orientation', 'Densely populated areas, <36° building orientation', 'Roads and sparse-moderately populated areas', 'Light vegetation',
        'Buildings surrounded by vegetation', 'Empty land', 'Water', 'Dark dense vegetation'] # labels for x-axis
y_axis_labels =  ['Densely populated areas, >36° building orientation', 'Densely populated areas, <36° building orientation', 'Roads and sparse-moderately populated areas', 'Light vegetation',
        'Buildings surrounded by vegetation', 'Empty land', 'Water', 'Dark dense vegetation']

fig = plt.figure(figsize=(22,17))
ax1 = plt.subplot2grid((20,21), (0,0), colspan=19, rowspan=19)
ax3 = plt.subplot2grid((20,21), (0,20), colspan=1, rowspan=19)

pv = pd.crosstab(dist.cl1, dist.cl2, values=dist['distances'], aggfunc='mean')

h1 = sns.heatmap(pv, xticklabels=x_axis_labels, yticklabels=y_axis_labels, cmap='YlGnBu',ax=ax1)
h2 = sns.heatmap(intra[['sdev_norm']], ax=ax3,  annot=False, cmap="YlGnBu", cbar=False, xticklabels=False, yticklabels=False)

ax1.set_title('Distance between centroids', pad=18, fontsize=20, fontweight='bold')
ax3.set_title('Average distance to centroid', pad=18, fontsize=20, fontweight='bold')

ax1.set(ylabel=None)
ax1.set(xlabel=None)
ax3.set(ylabel=None)

cbar = ax1.collections[0].colorbar
cbar.set_ticks([0, 0.25])
cbar.ax.set_ylabel('distance in the feature space', rotation=270)
cbar.set_ticklabels(['close', 'far'])
plt.tight_layout()
fig.savefig('../distance_features.png', transparent=False, facecolor='w', bbox_inches='tight')
plt.show()
