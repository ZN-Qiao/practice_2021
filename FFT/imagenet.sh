#!/bin/bash

python3 ./main_fft.py -a resnet50 -e --bw 3 --filter 0 ../../imagenet
python3 ./main_fft.py -a resnet50 -e --bw 6 --filter 0 ../../imagenet
python3 ./main_fft.py -a resnet50 -e --bw 9 --filter 0 ../../imagenet
python3 ./main_fft.py -a resnet50 -e --bw 12 --filter 0 ../../imagenet
python3 ./main_fft.py -a resnet50 -e --bw 15 --filter 0 ../../imagenet
python3 ./main_fft.py -a resnet50 -e --bw 18 --filter 0 ../../imagenet
python3 ./main_fft.py -a resnet50 -e --bw 21 --filter 0 ../../imagenet
python3 ./main_fft.py -a resnet50 -e --bw 24 --filter 0 ../../imagenet
python3 ./main_fft.py -a resnet50 -e --bw 27 --filter 0 ../../imagenet
python3 ./main_fft.py -a resnet50 -e --bw 30 --filter 0 ../../imagenet
python3 ./main_fft.py -a resnet50 -e --bw 3 --filter 1 ../../imagenet
python3 ./main_fft.py -a resnet50 -e --bw 6 --filter 1 ../../imagenet
python3 ./main_fft.py -a resnet50 -e --bw 9 --filter 1 ../../imagenet
python3 ./main_fft.py -a resnet50 -e --bw 12 --filter 1 ../../imagenet
python3 ./main_fft.py -a resnet50 -e --bw 15 --filter 1 ../../imagenet
python3 ./main_fft.py -a resnet50 -e --bw 18 --filter 1 ../../imagenet
python3 ./main_fft.py -a resnet50 -e --bw 21 --filter 1 ../../imagenet
python3 ./main_fft.py -a resnet50 -e --bw 24 --filter 1 ../../imagenet
python3 ./main_fft.py -a resnet50 -e --bw 27 --filter 1 ../../imagenet
python3 ./main_fft.py -a resnet50 -e --bw 30 --filter 1 ../../imagenet