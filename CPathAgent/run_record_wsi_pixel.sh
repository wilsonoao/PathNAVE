#!/bin/bash

python record_wsi_pixel.py /work/nas/WSI/01_BRCA \
    /work/project/data/TCGA-BRCA-FS/Tumor_annotation/BRCA

python record_wsi_pixel.py /work/nas/WSI/04_LUAD \
    /work/project/data/TCGA-LUAD-FS/Tumor_annotation/LUAD

python record_wsi_pixel.py /work/nas/WSI/10_LUSC \
    /work/project/data/TCGA-LUSC-FS/Tumor_annotation/LUSC
