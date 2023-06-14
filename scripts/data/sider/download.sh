#!/bin/bash

BASE_URL='http://sideeffects.embl.de/media/download/'
FILES=(
    'README'
    'meddra_all_indications.tsv.gz'
    'meddra_all_se.tsv.gz'
    'meddra_freq.tsv.gz'
)

SAVE_PATH="../../../data/databases/SIDER/"

for url in "${FILES[@]}"
do
  wget -P "${SAVE_PATH}" "${BASE_URL}${url}"  -O $url
done
