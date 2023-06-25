#change seed to produce multiple sets 
CUDA_VISIBLE_DEVICES=0,1 NUM_WORKERS=8 python main.py -s 1 --config ./configs/randocc.yaml --opts OUTPUT_PATH "./Result/RandOcc_seed1/"

