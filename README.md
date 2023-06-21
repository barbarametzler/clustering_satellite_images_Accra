# Clustering satellite images of Accra

Code used for the paper "Phenotyping urban built and natural environments with high-resolution
satellite images and unsupervised deep learning" published in STOTEN, 2023. \url{https://doi.org/10.1016/j.scitotenv.2023.164794}


The clustering algorithm was adapted from the DeepCluster algorithm by Caron (2018), which was published by Facebook research and
is also openly available, was run on 3 RTX6000 GPUs, 72GB memory and a runtime of approximately 24 h.

https://github.com/facebookresearch/deepcluster

## Requirements
- Python 3
- the SciPy and scikit-learn packages
- a PyTorch install version 0.1.8 (pytorch.org)
- CUDA 8.0
- a Faiss install (Faiss)
- The ImageNet dataset (which can be automatically downloaded by recent version of torchvision)



## Data sources
Building information: https://ui.adsabs.harvard.edu/abs/2019AGUFMIN11D0688H/abstract. The building information is provided in a vector format. We overlaid the vector with a grid that
represents the tile size and location. For each measure, we calculated the mean value per tile. Building orientation was computed with the momepy package (Fleischmann, 2019) as deviation of orientation from cardinal directions; it was defined as an orientation of the longest axis of the bounding rectangle in range 0–45 degrees. Building orientation is measured with respect to cardinal directions Building orientation also has a physical relevance for residents, impacting the natural lighting and ventilation.

Road data: https://www.openstreetmap.org/. The road information is provided in a vector format. We overlaid the vector with a grid that represents the tile size and location, and
calculated statistics per tile.

Population density: https://www.worldpop.org/geodata/summary?id=6116. We used a population raster with a resolution of 100 m to calculate the mean population density per tile. It was computed by vectorising the population density raster file, overlaying it with the tile grid and calculating the mean per tile.

NDVI: https://www.usgs.gov/centers/eros/science/usgs-eros-archive-landsat-archives-landsat-8-oli-operational-land-imager-and. We use Landsat imagery from 01/01/2020,
a cloudless day, to calculate the mean NDVI value for each tile.
