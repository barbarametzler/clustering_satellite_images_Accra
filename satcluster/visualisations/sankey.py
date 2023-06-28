## A Barbara Metzler
## code for sankey plots

# import libraries
import pandas as pd
import geopandas as gp
from pysankey2 import Sankey
import seaborn as sns

import matplotlib.pyplot as plt


cls_map = {'Dark dense vegetation': '#047304', 'Other': '#cdc9c4', 'Dark dense vegetation and water': '#047304', 'Populated areas': '#858256', 'Empty land and light vegetation': '#a35d0a',
          'Water': '#497bbb', 'Densely populated areas, <36° orientation': '#757471','Densely populated areas': '#757471','Populated areas and roads': '#858256', 'Densely populated areas, >36° orientation': '#8d8d8d',
           'Empty land': '#a35d0a', 'Roads and sparse-moderately populated': '#cdc9c4','Buildings surrounded by vegetation': '#858256', 'Light vegetation': '#8abc47',
           'Densely populated, >36° orientation': '#757471', 'Mixed environments': '#313a2e', 'Densely populated, 11-36° orientation':'#f3d094',
          'Densely populated, 28-36° orientation': '#c8bc8c', 'Densely populated, 10-27° orientation': '#f3d194', 'Empty or sandy land': '#a35d0a',
      'Edges of clouds': '#94e8f0', 'Riperian areas': '#307168', 'Urban vegetation': '#8aba48', 'Densely populated, <10° orientation': '#7b7b7d'}


# input pandas DataFrame with columns that represent cluster labels at certain K
multidiagram = Sankey(df[['label_k2', 'label_k4', 'label_k6', 'label_new', 'label_k10', 'label_k12']],colorDict=cls_map,  colorMode="global", stripColor='left')
fig,ax = multidiagram.plot(figSize=(45, 20), fontSize=18)
fig.savefig('../sankey.png', transparent=False, facecolor='w', bbox_inches='tight')

plt.show()
