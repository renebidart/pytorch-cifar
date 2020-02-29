CUDA_VISIBLE_DEVICES=1 python main.py --model_type albert --n_layers 1 --n_hidden 256 --norm batch --spatial conv --save_path ./results/albert_nh256_nl1_Sconv_Nbatch
CUDA_VISIBLE_DEVICES=1 python main.py --model_type albert --n_layers 4 --n_hidden 256 --norm batch --spatial conv --save_path ./results/albert_nh256_nl4_Sconv_Nbatch

CUDA_VISIBLE_DEVICES=1 python main.py --model_type albert --n_layers 1 --n_hidden 256 --norm layer --spatial conv --save_path ./results/albert_nh256_nl1_Sconv_Nlayer
CUDA_VISIBLE_DEVICES=1 python main.py --model_type albert --n_layers 4 --n_hidden 256 --norm layer --spatial conv --save_path ./results/albert_nh256_nl4_Sconv_Nlayer

CUDA_VISIBLE_DEVICES=1 python main.py --model_type albert --n_layers 1 --n_hidden 256 --norm layer_noaffine --spatial conv --save_path ./results/albert/albert_nh256_nl1_Sconv_Nlayer_noaffine
CUDA_VISIBLE_DEVICES=1 python main.py --model_type albert --n_layers 4 --n_hidden 256 --norm layer_noaffine --spatial conv --save_path ./results/albert/albert_nh256_nl4_Sconv_Nlayer_noaffine

CUDA_VISIBLE_DEVICES=1 python main.py --model_type albert --n_layers 1 --n_hidden 256 --norm group --spatial conv --save_path ./results/albert_nh256_nl1_Sconv_Ngroup
CUDA_VISIBLE_DEVICES=1 python main.py --model_type albert --n_layers 4 --n_hidden 256 --norm group --spatial conv --save_path ./results/albert_nh256_nl4_Sconv_Ngroup

CUDA_VISIBLE_DEVICES=1 python main.py --model_type albert --n_layers 1 --n_hidden 256 --norm _instance --spatial conv --save_path ./results/albert_nh256_nl1_Sconv_Ninstance
CUDA_VISIBLE_DEVICES=1 python main.py --model_type albert --n_layers 4 --n_hidden 256 --norm _instance --spatial conv --save_path ./results/albert_nh256_nl4_Sconv_Ninstance