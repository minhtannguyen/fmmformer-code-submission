## Code for Language Modeling Task in Our Paper

## Requirements
This toolkit requires PyTorch `torch` and Ninja `ninja` (to compile the cuda kernels).

The experiments for the paper were conducted with Python 3.6 and PyTorch >= 1.4.0.

The toolkit supports [Weights & Biases](https://docs.wandb.ai/) for monitoring jobs. If you use it, also install `wandb`.

## Instructions

Run `sh getdata.sh` to download the data.

Run following commands to reproduce results in Table 2 and 3 in our paper.

Softmax
```
CUDA_VISIBLE_DEVICES=0,1 python ./src/train.py --cuda --data ./data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 2 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --multi_gpu --use_wandb --project_name 'fmmformer' --job_name 'softmax' --work_dir ./softmax
```

Linear
```
CUDA_VISIBLE_DEVICES=0,1 python ./src/train.py --cuda --data ./data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 34 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --multi_gpu --use_wandb --project_name 'fmmformer' --job_name 'linear' --work_dir ./linear
```

Band_5
```
CUDA_VISIBLE_DEVICES=0,1 python ./src/train.py --cuda --data ./data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 5 --sparse_ratio 4.5 --kernels 'elu' --multi_gpu --use_wandb --project_name 'fmmformer' --job_name 'fmmformer-band-5' --work_dir ./fmmformer-band-5
```

Band_20
```
CUDA_VISIBLE_DEVICES=0,1 python ./src/train.py --cuda --data ./data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 20 --sparse_ratio 4.5 --kernels 'elu' --multi_gpu --use_wandb --project_name 'fmmformer' --job_name 'fmmformer-band-20' --work_dir ./fmmformer-band-20
```

FMMformer (1-kernel + Band_5)
```
CUDA_VISIBLE_DEVICES=0,1 python ./src/train.py --cuda --data ./data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 5 --sparse_ratio 2.5 --kernels 'elu' --multi_gpu --use_wandb --project_name 'fmmformer' --job_name 'fmmformer-1-kernel-band-5' --work_dir ./fmmformer-1-kernel-band-5
```

FMMformer (1-kernel + Band_20)
```
CUDA_VISIBLE_DEVICES=0,1 python ./src/train.py --cuda --data ./data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 20 --sparse_ratio 2.5 --kernels 'elu' --multi_gpu --use_wandb --project_name 'fmmformer' --job_name 'fmmformer-1-kernel-band-20' --work_dir ./fmmformer-1-kernel-band-20
```

FMMformer (2-kernel + Band_20)
```
CUDA_VISIBLE_DEVICES=0,1 python ./src/train.py --cuda --data ./data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 20 --sparse_ratio 2.5 --kernels 'elu' 'elu_flip' --multi_gpu --use_wandb --project_name 'fmmformer' --job_name 'fmmformer-2-kernel-band-20' --work_dir ./fmmformer-2-kernel-band-20
```
