#!/bin/bash

#PBS -lwalltime=72:00:00
#PBS -lselect=1:ncpus=32:mem=72gb:ngpus=3:gpu_type=RTX6000

DIR="~/ssa_satellite_imagery/live/Raster_data/Accra/"
ARCH="vgg16" #or alexnet/vgg16
LR=0.0001
WD=-5
CLUSTERING=Kmeans #Kmeans
K=16 #10000
WORKERS=4
EXP="~/exp/"
BATCH=128
RESUME="~/checkpoint.pth.tar"

FEPOCH=20
FNAME="/features_"
mkdir -p ${EXP}

module load anaconda3/personal

#CUDA_VISIBLE_DEVICES=0 
python3 /rds/general/user/abm1818/home/GitHub/deepcluster-master/main_sfeatures.py ${DIR} --exp ${EXP} --arch ${ARCH} \
  --lr ${LR} --wd ${WD} --k ${K} --sobel --clustering ${CLUSTERING} --verbose --batch ${BATCH} --workers ${WORKERS} --resume ${RESUME} --features_name ${FNAME} --features_epoch ${FEPOCH}


