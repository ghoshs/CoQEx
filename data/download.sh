#!/bin/bash

COQEX_ROOT=$(pwd)

echo 'root dir:' $COQEX_ROOT

# SIGIR data

echo "Creating a dataset dir if necessary .."
mkdir -p ${COQEX_ROOT}/dataset/SIGIR22/

echo "Downloading CoQuAD train and evaluation data .."
wget -r -O coquad.zip https://nextcloud.mpi-klsb.mpg.de/index.php/s/kiWC5wEgR9nxws5/download
unzip coquad.zip -d ${COQEX_ROOT}/dataset/SIGIR22/
rm coquad.zip
echo "Download complete!"

echo "Downloading count query subset from LCQuAD v2 .."
wget -r -O colcquad.zip https://nextcloud.mpi-klsb.mpg.de/index.php/s/9eDTnArAJmyLwkd/download
unzip colcquad.zip -d ${COQEX_ROOT}/dataset/SIGIR22/
rm colcquad.zip
echo "Download complete!"

echo "Downloading count query Stresstest .."
wget -r -O stresstest.zip https://nextcloud.mpi-klsb.mpg.de/index.php/s/G5NfyMo4DARJGAF/download
unzip stresstest.zip -d ${COQEX_ROOT}/dataset/SIGIR22/
rm stresstest.zip
echo "Download complete!"

# JoWS data

echo "Creating a dataset dir if necessary .."
mkdir -p ${COQEX_ROOT}/dataset/JoWS22/

echo "Downloading CoQuADv2 train and evaluation data .."
wget -r -O coquadv2.zip https://nextcloud.mpi-klsb.mpg.de/index.php/s/Qp92Nc9irdKDZre/download
unzip coquadv2.zip -d ${COQEX_ROOT}/dataset/JoWS22/
rm coquadv2.zip
echo "Download complete!"

echo "Downloading count query subset from Natural Questions .."
wget -r -O conq.zip https://nextcloud.mpi-klsb.mpg.de/index.php/s/HRd6L9XRyYxZ8rW/download
unzip conq.zip -d ${COQEX_ROOT}/dataset/JoWS22/
rm conq.zip
echo "Download complete!"