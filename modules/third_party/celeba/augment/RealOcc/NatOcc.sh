
# NatOcc hands without SOT
#CUDA_VISIBLE_DEVICES=0,1 NUM_WORKERS=8 python main.py --config ./configs/natocc_hand.yaml 

# NatOcc hands with SOT
CUDA_VISIBLE_DEVICES=0 NUM_WORKERS=4 python main.py --config ./configs/natocc_hand.yaml --opts OUTPUT_PATH "./Result/NatOcc_hand_sot/" AUGMENTATION.SOT True 
 
# NatOcc objects
#CUDA_VISIBLE_DEVICES=0,1 NUM_WORKERS=8 python main.py --config ./configs/natocc_objects.yaml

