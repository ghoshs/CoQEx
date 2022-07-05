#!/bin/bash

COQEX_ROOT=$(pwd)

echo 'root dir:' $COQEX_ROOT

echo "Creating a dataset dir if necessary .."
mkdir -p ${COQEX_ROOT}/dataset/

echo "Downloading CoQuAD_v1 data .."
wget -r -O coquad.zip https://nextcloud.mpi-klsb.mpg.de/index.php/s/kiWC5wEgR9nxws5/download
unzip coquad.zip -d ${COQEX_ROOT}/dataset/
rm coquad.zip
echo "Download complete!"

echo "Downloading count query subset from LCQuAD v2 .."
wget -r -O colcquad.zip https://nextcloud.mpi-klsb.mpg.de/index.php/s/9eDTnArAJmyLwkd/download
unzip colcquad.zip -d ${COQEX_ROOT}/dataset/
rm colcquad.zip
echo "Download complete!"

echo "Downloading count query Stresstest .."
wget -r -O stresstest.zip https://nextcloud.mpi-klsb.mpg.de/index.php/s/G5NfyMo4DARJGAF/download
unzip stresstest.zip -d ${COQEX_ROOT}/dataset/
rm stresstest.zip
echo "Download complete!"