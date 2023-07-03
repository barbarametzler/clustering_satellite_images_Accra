# Clustering satellite images of Accra

Code used for the paper "Phenotyping urban built and natural environments with high-resolution
satellite images and unsupervised deep learning" published in STOTEN, 2023. Access here: https://doi.org/10.1016/j.scitotenv.2023.164794


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


## Usage

as defined in satcluster/DeepCluster/example_script.sh

```
main_sfeatures.py [-h] [--arch ARCH] [--sobel] [--clustering {Kmeans,PIC}]
               [--nmb_cluster NMB_CLUSTER] [--lr LR] [--wd WD]
               [--reassign REASSIGN] [--workers WORKERS] [--epochs EPOCHS]
               [--start_epoch START_EPOCH] [--batch BATCH]
               [--momentum MOMENTUM] [--resume PATH]
               [--checkpoints CHECKPOINTS] [--seed SEED] [--exp EXP] [--features_epoch] [--features_epoch]
               [--verbose]
               DIR

positional arguments:
  DIR                   path to dataset

optional arguments:
  -h, --help            show this help message and exit
  --arch ARCH, -a ARCH11CNN architecture (default: vgg16)
  --sobel               Sobel filtering
  --clustering {Kmeans,PIC}
                        clustering algorithm (default: Kmeans)
  --nmb_cluster NMB_CLUSTER, --k NMB_CLUSTER
                        number of cluster for k-means (default: 8)
  --lr LR               learning rate (default: 0.0001)
  --wd WD               weight decay pow (default: -5)
  --reassign REASSIGN   how many epochs of training between two consecutive
                        reassignments of clusters (default: 1)
  --workers WORKERS     number of data loading workers (default: 4)
  --epochs EPOCHS       number of total epochs to run (default: 20)
  --start_epoch START_EPOCH
                        manual epoch number (useful on restarts) (default: 0)
  --batch BATCH         mini-batch size (default: 256)
  --momentum MOMENTUM   momentum (default: 0.9)
  --resume PATH         path to checkpoint (default: None)
  --checkpoints CHECKPOINTS
                        how many iterations between two checkpoints (default:
                        25000)
  --seed SEED           random seed (default: 31)
  --exp EXP             path to exp folder
  --features_name       name of image representations to be saved
  --features_epoch      epoch at which image representations are saved (default: 20)
  --verbose             chatty
```



## Visualisations

code for visualisations is added in 'visualisations' folder.

### Requirements for visualisations
- Python 3
- SciPy, pandas and scikit-learn packages
- Plotting: Seaborn and matplotlib
- Sankey diagram: install pySankey2 from https://github.com/SZJShuffle/pySankey2


## Data sources
Building information: https://ui.adsabs.harvard.edu/abs/2019AGUFMIN11D0688H/abstract. The building information is provided in a vector format. We overlaid the vector with a grid that
represents the tile size and location. For each measure, we calculated the mean value per tile. Building orientation was computed with the momepy package (Fleischmann, 2019) as deviation of orientation from cardinal directions; it was defined as an orientation of the longest axis of the bounding rectangle in range 0–45 degrees. Building orientation is measured with respect to cardinal directions Building orientation also has a physical relevance for residents, impacting the natural lighting and ventilation.

Road data: https://www.openstreetmap.org/. The road information is provided in a vector format. We overlaid the vector with a grid that represents the tile size and location, and
calculated statistics per tile.

Population density: https://www.worldpop.org/geodata/summary?id=6116. We used a population raster with a resolution of 100 m to calculate the mean population density per tile. It was computed by vectorising the population density raster file, overlaying it with the tile grid and calculating the mean per tile.

NDVI: https://www.usgs.gov/centers/eros/science/usgs-eros-archive-landsat-archives-landsat-8-oli-operational-land-imager-and. We use Landsat imagery from 01/01/2020,
a cloudless day, to calculate the mean NDVI value for each tile.


## Final cluster shapefile
The final cluster shapefile (cluster_shapefile.geojson)is saved as a Geopandas DataFrame with CRS = 'EPSG:32630', which translate to
WGS 84 / UTM zone 30N.

The Python dictionary for mapping cluster labels to cluster numbers is saved in vis_util.py
