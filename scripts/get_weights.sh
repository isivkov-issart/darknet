#!/bin/bash

mkdir weights
cd weights

wget 'https://pjreddie.com/media/files/yolov2.weights'
wget 'https://pjreddie.com/media/files/yolov2-tiny.weights'
wget 'https://pjreddie.com/media/files/yolov3.weights'
wget 'https://pjreddie.com/media/files/yolov3-tiny.weights'
wget 'https://pjreddie.com/media/files/yolov3-spp.weights'

cd ..

