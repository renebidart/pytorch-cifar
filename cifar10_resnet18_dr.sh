
CUDA_VISIBLE_DEVICES=1 python main.py --num_classes 10 --model_type resnet18 --ws_factor 2 --downsample_repeat --channel_factor 1 --save_path ./saved_models/cifar10/resnet18_ws2_ch1_dr
CUDA_VISIBLE_DEVICES=1 python main.py --num_classes 10 --model_type resnet18 --ws_factor 4 --downsample_repeat --channel_factor 1 --save_path ./saved_models/cifar10/resnet18_ws4_ch1_dr
CUDA_VISIBLE_DEVICES=1 python main.py --num_classes 10 --model_type resnet18 --ws_factor 8 --downsample_repeat --channel_factor 1 --save_path ./saved_models/cifar10/resnet18_ws8_ch1_dr
CUDA_VISIBLE_DEVICES=1 python main.py --num_classes 10 --model_type resnet18 --ws_factor 16 --downsample_repeat --channel_factor 1 --save_path ./saved_models/cifar10/resnet18_ws16_ch1_dr
