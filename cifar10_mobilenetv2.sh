CUDA_VISIBLE_DEVICES=0 python main.py --model_type mobilenetv2  --save_path ./saved_models/cifar10/mobile --bs 128 --lr .1
CUDA_VISIBLE_DEVICES=0 python main.py --model_type wsmobilenetv2 --ws_sp_factor 1  --save_path ./saved_models/cifar10/mobile_wss1 --bs 256 --lr .2
CUDA_VISIBLE_DEVICES=0 python main.py  --model_type wsmobilenetv2 --ws_sp_factor 2  --save_path ./saved_models/cifar10/mobile_wss2 --bs 256 --lr .2 
CUDA_VISIBLE_DEVICES=0 python main.py  --model_type wsmobilenetv2 --ws_sp_factor 4  --save_path ./saved_models/cifar10/mobile_wss4 --bs 256 --lr .2
CUDA_VISIBLE_DEVICES=0 python main.py --model_type wsmobilenetv2 --ws_sp_factor 8  --save_path ./saved_models/cifar10/mobile_wss8 --bs 256 --lr .2

# CUDA_VISIBLE_DEVICES=1 python main.py --model_type resnet18std  --save_path ./saved_models/cifar10/resnet18std --bs 128 --lr .1
