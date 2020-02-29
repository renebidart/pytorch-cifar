# testing on 1 and 4 loops to see if looping helps (it better)
# comparing conv (7x7) with 2d self attention
# Trying a bunch of different norms. Most promising 3 for now, others later

### ATTENTION MODELS
CUDA_VISIBLE_DEVICES=1 python main.py --model_type albert --n_layers 1 --n_hidden 256 --norm batch --spatial attn --save_path ./results/albert_nh256_nl1_Sattn_Nbatch --bs 128 --lr .05
CUDA_VISIBLE_DEVICES=1 python main.py --model_type albert --n_layers 4 --n_hidden 256 --norm batch --spatial attn --save_path ./results/albert_nh256_nl4_Sattn_Nbatch --bs 128 --lr .05

CUDA_VISIBLE_DEVICES=1 python main.py --model_type albert --n_layers 1 --n_hidden 256 --norm layer_albert --spatial attn --save_path ./results/albert_nh256_nl1_Sattn_Nlayer_albert --bs 128 --lr .05
CUDA_VISIBLE_DEVICES=1 python main.py --model_type albert --n_layers 4 --n_hidden 256 --norm layer_albert --spatial attn --save_path ./results/albert_nh256_nl4_Sattn_Nlayer_albert --bs 128 --lr .05

CUDA_VISIBLE_DEVICES=1 python main.py --model_type albert --n_layers 1 --n_hidden 256 --norm group --spatial attn --save_path ./results/albert_nh256_nl1_Sattn_Ngroup --bs 128 --lr .05
CUDA_VISIBLE_DEVICES=1 python main.py --model_type albert --n_layers 4 --n_hidden 256 --norm group --spatial attn --save_path ./results/albert_nh256_nl4_Sattn_Ngroup --bs 128 --lr .05

# CUDA_VISIBLE_DEVICES=1 python main.py --model_type albert --n_layers 1 --n_hidden 256 --norm layer --spatial attn --save_path ./results/albert/albert_nh256_nl1_Sattn_Nlayer  --bs 128 --lr .05 
# CUDA_VISIBLE_DEVICES=1 python main.py --model_type albert --n_layers 4 --n_hidden 256 --norm layer --spatial attn --save_path ./results/albert/albert_nh256_nl4_Sattn_Nlayer  --bs 128 --lr .05

# CUDA_VISIBLE_DEVICES=1 python main.py --model_type albert --n_layers 1 --n_hidden 256 --norm layer_noaffine --spatial attn --save_path ./results/albert/albert_nh256_nl1_Sattn_Nlayer_noaffine  --bs 128 --lr .05
# CUDA_VISIBLE_DEVICES=1 python main.py --model_type albert --n_layers 4 --n_hidden 256 --norm layer_noaffine --spatial attn --save_path ./results/albert/albert_nh256_nl4_Sattn_Nlayer_noaffine  --bs 128 --lr .05

# CUDA_VISIBLE_DEVICES=1 python main.py --model_type albert --n_layers 1 --n_hidden 256 --norm instance --spatial attn --save_path ./results/albert_nh256_nl1_Sattn_Ninstance  --bs 128 --lr .05
# CUDA_VISIBLE_DEVICES=1 python main.py --model_type albert --n_layers 4 --n_hidden 256 --norm instance --spatial attn --save_path ./results/albert_nh256_nl4_Sattn_Ninstance  --bs 128 --lr .05


# CONV MODELS
CUDA_VISIBLE_DEVICES=1 python main.py --model_type albert --n_layers 1 --n_hidden 256 --norm batch --spatial conv --save_path ./results/albert_nh256_nl1_Sconv_Nbatch
CUDA_VISIBLE_DEVICES=1 python main.py --model_type albert --n_layers 4 --n_hidden 256 --norm batch --spatial conv --save_path ./results/albert_nh256_nl4_Sconv_Nbatch

CUDA_VISIBLE_DEVICES=1 python main.py --model_type albert --n_layers 1 --n_hidden 256 --norm layer_albert --spatial conv --save_path ./results/albert_nh256_nl1_Sconv_Nlayer_albert
CUDA_VISIBLE_DEVICES=1 python main.py --model_type albert --n_layers 4 --n_hidden 256 --norm layer_albert --spatial conv --save_path ./results/albert_nh256_nl4_Sconv_Nlayer_albert

CUDA_VISIBLE_DEVICES=1 python main.py --model_type albert --n_layers 1 --n_hidden 256 --norm group --spatial conv --save_path ./results/albert_nh256_nl1_Sconv_Ngroup
CUDA_VISIBLE_DEVICES=1 python main.py --model_type albert --n_layers 4 --n_hidden 256 --norm group --spatial conv --save_path ./results/albert_nh256_nl4_Sconv_Ngroup

# CUDA_VISIBLE_DEVICES=1 python main.py --model_type albert --n_layers 1 --n_hidden 256 --norm layer --spatial conv --save_path ./results/albert/albert_nh256_nl1_Sconv_Nlayer
# CUDA_VISIBLE_DEVICES=1 python main.py --model_type albert --n_layers 4 --n_hidden 256 --norm layer --spatial conv --save_path ./results/albert/albert_nh256_nl4_Sconv_Nlayer

# CUDA_VISIBLE_DEVICES=1 python main.py --model_type albert --n_layers 1 --n_hidden 256 --norm layer_noaffine --spatial conv --save_path ./results/albert/albert_nh256_nl1_Sconv_Nlayer_noaffine
# CUDA_VISIBLE_DEVICES=1 python main.py --model_type albert --n_layers 4 --n_hidden 256 --norm layer_noaffine --spatial conv --save_path ./results/albert/albert_nh256_nl4_Sconv_Nlayer_noaffine

# CUDA_VISIBLE_DEVICES=1 python main.py --model_type albert --n_layers 1 --n_hidden 256 --norm instance --spatial conv --save_path ./results/albert_nh256_nl1_Sconv_Ninstance
# CUDA_VISIBLE_DEVICES=1 python main.py --model_type albert --n_layers 4 --n_hidden 256 --norm instance --spatial conv --save_path ./results/albert_nh256_nl4_Sconv_Ninstance


