#!/bin/bash

set -e

mkdir data/lp-detector -p
mkdir data/ocr -p
mkdir data/vehicle-detector -p
mkdir data/FRD -p

wget -c -N www.inf.ufrgs.br/~smsilva/alpr-unconstrained/data/lp-detector/wpod-net_update1.h5   -P data/lp-detector/
wget -c -N www.inf.ufrgs.br/~smsilva/alpr-unconstrained/data/lp-detector/wpod-net_update1.json -P data/lp-detector/

wget -c -N www.inf.ufrgs.br/~smsilva/alpr-unconstrained/data/ocr/ocr-net.cfg     -P data/ocr/
wget -c -N www.inf.ufrgs.br/~smsilva/alpr-unconstrained/data/ocr/ocr-net.names   -P data/ocr/
wget -c -N www.inf.ufrgs.br/~smsilva/alpr-unconstrained/data/ocr/ocr-net.weights -P data/ocr/
wget -c -N www.inf.ufrgs.br/~smsilva/alpr-unconstrained/data/ocr/ocr-net.data    -P data/ocr/

wget -c -N www.inf.ufrgs.br/~smsilva/alpr-unconstrained/data/vehicle-detector/yolo-voc.cfg     -P data/vehicle-detector/
wget -c -N www.inf.ufrgs.br/~smsilva/alpr-unconstrained/data/vehicle-detector/voc.data         -P data/vehicle-detector/
wget -c -N www.inf.ufrgs.br/~smsilva/alpr-unconstrained/data/vehicle-detector/yolo-voc.weights -P data/vehicle-detector/
wget -c -N www.inf.ufrgs.br/~smsilva/alpr-unconstrained/data/vehicle-detector/voc.names        -P data/vehicle-detector/

wget -c -N harmony.cs.pusan.ac.kr/~nrlab/2019MITA/Vehicle-Front-Rear/FRNet.names                      -P data/FRD/
wget -c -N harmony.cs.pusan.ac.kr/~nrlab/2019MITA/Vehicle-Front-Rear/FRNet_YOLOv3.cfg                 -P data/FRD/
wget -c -N harmony.cs.pusan.ac.kr/~nrlab/2019MITA/Vehicle-Front-Rear/FRNet_YOLOv3.data                -P data/FRD/
wget -c -N harmony.cs.pusan.ac.kr/~nrlab/2019MITA/Vehicle-Front-Rear/FRNet_YOLOv3_50000.weights       -P data/FRD/
wget -c -N harmony.cs.pusan.ac.kr/~nrlab/2019MITA/Vehicle-Front-Rear/FRNet_YOLOv3_tiny.cfg            -P data/FRD/
wget -c -N harmony.cs.pusan.ac.kr/~nrlab/2019MITA/Vehicle-Front-Rear/FRNet_YOLOv3_tiny.data           -P data/FRD/
wget -c -N harmony.cs.pusan.ac.kr/~nrlab/2019MITA/Vehicle-Front-Rear/FRNet_YOLOv3_tiny_126000.weights -P data/FRD/

