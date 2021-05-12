CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 nohup python3 cifar_normal.py --netName=resnet20 &
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 nohup python3 cifar_sign.py --netName=resnet110 &
