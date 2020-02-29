CUDA_VISIBLE_DEVICES=0 python main.py --num_classes 10 --model_type resnet18bn --ws_factor 1 --channel_factor 1 --save_path ./saved_models/cifar10/resnet18bn_ws1_ch1

CUDA_VISIBLE_DEVICES=0 python main.py --num_classes 10 --model_type resnet18bn --ws_factor 2 --channel_factor 1 --save_path ./saved_models/cifar10/resnet18bn_ws2_ch1

CUDA_VISIBLE_DEVICES=0 python main.py --num_classes 10 --model_type resnet18bn --ws_factor 4 --channel_factor 1 --save_path ./saved_models/cifar10/resnet18bn_ws4_ch1

CUDA_VISIBLE_DEVICES=0 python main.py --num_classes 10 --model_type resnet18bn --ws_factor 8 --channel_factor 1 --save_path ./saved_models/cifar10/resnet18bn_ws8_ch1

CUDA_VISIBLE_DEVICES=0 python main.py --num_classes 10 --model_type resnet18bn --ws_factor 16 --channel_factor 1 --save_path ./saved_models/cifar10/resnet18bn_ws16_ch1


CUDA_VISIBLE_DEVICES=0 python main.py --num_classes 10 --model_type resnet18bn --ws_factor 1 --channel_factor 2 --save_path ./saved_models/cifar10/resnet18bn_ws1_ch2

CUDA_VISIBLE_DEVICES=0 python main.py --num_classes 10 --model_type resnet18bn --ws_factor 1 --channel_factor 4 --save_path ./saved_models/cifar10/resnet18bn_ws1_ch4

CUDA_VISIBLE_DEVICES=0 python main.py --num_classes 10 --model_type resnet18bn --ws_factor 1 --channel_factor 8 --save_path ./saved_models/cifar10/resnet18bn_ws1_ch8

CUDA_VISIBLE_DEVICES=0 python main.py --num_classes 10 --model_type resnet18bn --ws_factor 1 --channel_factor 16 --save_path ./saved_models/cifar10/resnet18bn_ws1_ch16