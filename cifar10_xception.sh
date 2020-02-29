CUDA_VISIBLE_DEVICES=1 python main.py --model_type xception --channel_factor 1 --ws_sp_factor 1 --ws_ch_factor 1 --save_path ./saved_models/cifar10/xception_ch1_wss_1_wsc_1 --bs 256 --lr .1
CUDA_VISIBLE_DEVICES=1 python main.py  --model_type xception --channel_factor 2 --ws_sp_factor 1 --ws_ch_factor 1 --save_path ./saved_models/cifar10/xception_ch2_wss_1_wsc_1 --bs 256 --lr .1 
CUDA_VISIBLE_DEVICES=1 python main.py  --model_type xception --channel_factor 4 --ws_sp_factor 1 --ws_ch_factor 1 --save_path ./saved_models/cifar10/xception_ch4_wss_1_wsc_1 --bs 256 --lr .1
CUDA_VISIBLE_DEVICES=1 python main.py --model_type xception --channel_factor 8 --ws_sp_factor 1 --ws_ch_factor 1 --save_path ./saved_models/cifar10/xception_ch8_wss_1_wsc_1 --bs 256 --lr .1

C56A_VISIBLE_DEVICES=1 python main.py  --model_type xception --channel_factor 1 --ws_sp_factor 2 --ws_ch_factor 1 --save_path ./saved_models/cifar10/xception_ch1_wss_2_wsc_1 --bs 256 --lr .1
# CUDA_VISIBLE_DEVICES=1 python main.py  --model_type xception --channel_factor 1 --ws_sp_factor 4 --ws_ch_factor 1 --save_path ./saved_models/cifar10/xception_ch1_wss_4_wsc_1 --bs 256 --lr .1
CUDA_VISIBLE_DEVICES=1 python main.py --model_type xception --channel_factor 1 --ws_sp_factor 8 --ws_ch_factor 1 --save_path ./saved_models/cifar10/xception_ch1_wss_8_wsc_1 --bs 256 --lr .1

CUDA_VISIBLE_DEVICES=1 python main.py  --model_type xception --channel_factor 1 --ws_sp_factor 1 --ws_ch_factor 2 --save_path ./saved_models/cifar10/xception_ch1_wss_1_wsc_2 --bs 256 --lr .1
CUDA_VISIBLE_DEVICES=1 python main.py  --model_type xception --channel_factor 1 --ws_sp_factor 1 --ws_ch_factor 4 --save_path ./saved_models/cifar10/xception_ch1_wss_1_wsc_4 --bs 256 --lr .1
CUDA_VISIBLE_DEVICES=1 python main.py --model_type xception --channel_factor 1 --ws_sp_factor 1 --ws_ch_factor 8 --save_path ./saved_models/cifar10/xception_ch1_wss_1_wsc_8 --bs 256 --lr .1

# Both WS
CUDA_VISIBLE_DEVICES=1 python main.py --model_type xception --channel_factor 1 --ws_sp_factor 2 --ws_ch_factor 2 --save_path ./saved_models/cifar10/xception_ch1_wss_2_wsc_2 --bs 256 --lr .1
CUDA_VISIBLE_DEVICES=1 python main.py --model_type xception --channel_factor 1 --ws_sp_factor 4 --ws_ch_factor 4 --save_path ./saved_models/cifar10/xception_ch1_wss_4_wsc_4 --bs 256 --lr .1

# Both WS and CH reduction 
CUDA_VISIBLE_DEVICES=1 python main.py  --model_type xception --channel_factor 2 --ws_sp_factor 2 --ws_ch_factor 2 --save_path ./saved_models/cifar10/xception_ch2_wss_2_wsc_2 --bs 256 --lr .1
CUDA_VISIBLE_DEVICES=1 python main.py  --model_type xception --channel_factor 4 --ws_sp_factor 4 --ws_ch_factor 4 --save_path ./saved_models/cifar10/xception_ch4_wss_4_wsc_4 --bs 256 --lr .1
