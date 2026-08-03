#!/bin/bash

XML_DIR=./annotations
BASE_URL="s3://camelyon-dataset/CAMELYON16/images"
OUT_DIR=./slides

mkdir -p $OUT_DIR

for file in $XML_DIR/test_*.xml; do
    name=$(basename "$file" .xml)

    echo "Downloading $name.tif ..."

    aws s3 cp --no-sign-request \
    "${BASE_URL}/${name}.tif" \
    "${OUT_DIR}/${name}.tif"
done