#!/bin/bash

# # Linear Transformer with our fast weight memory update rule
# # --attn_type 24: indicates the combination: linear transformer + our update rule.

# # other options are:
# # Standard Transformer: 2

# # Standard models (Fast weights with sum update:
# #   * Linear Transformer: 4 (pure pytorch) or 34 (custom cuda kernel)
# #   * Performer: 5 (pure pytorch) or 35 (custom cuda kernel)
# #   * DPFP: 6 (pure pytorch) or 36 (custom cuda kernel)

# # Fast weights with our update rule:
# #   * Linear Transformer: 24 (no attn normalisation to retrieve with key) or 44 (with attn normalisation)
# #   * Performer: 25 (no attn normalisation to retrieve with key) or 45 (with attn normalisation)
# #   * DPFP: 26 (no attn normalisation to retrieve with key) or 46 (with attn normalisation)

# # for Performers, `m` can be specified via --performer_proj_dim
# # for DPFP, `\nu` can be specified via --dpfp_n_roll 

# # --skip_attn_normalization: disable the attn normalisation everywhere.
# # (this should give the best performance)

# ####################################################################


CUDA_VISIBLE_DEVICES=0,1 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 1 --sparse_ratio 0.5 --kernels 'tanh' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-tanh-attention-diag-1-spr-1w-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-tanh-attention-diag-1-spr-1w-final

CUDA_VISIBLE_DEVICES=2,3 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 1 --sparse_ratio 1.5 --kernels 'tanh' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-tanh-attention-diag-1-spr-2w-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-tanh-attention-diag-1-spr-2w-final

CUDA_VISIBLE_DEVICES=4,5 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 1 --sparse_ratio 2.5 --kernels 'tanh' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-tanh-attention-diag-1-spr-1sw-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-tanh-attention-diag-1-spr-1sw-final

CUDA_VISIBLE_DEVICES=6,7 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 1 --sparse_ratio 3.5 --kernels 'tanh' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-tanh-attention-diag-1-spr-2sw-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-tanh-attention-diag-1-spr-2sw-final

CUDA_VISIBLE_DEVICES=2,3 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 1 --sparse_ratio 4.5 --kernels 'tanh' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-tanh-attention-diag-1-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-tanh-attention-diag-1-final

CUDA_VISIBLE_DEVICES=0,1 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 1 --sparse_ratio 5.5 --kernels 'tanh' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-tanh-attention-lowrank-only-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-tanh-attention-lowrank-only-final







CUDA_VISIBLE_DEVICES=2,3 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 2 --sparse_ratio 0.5 --kernels 'tanh' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-tanh-attention-diag-2-spr-1w-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-tanh-attention-diag-2-spr-1w-final

CUDA_VISIBLE_DEVICES=4,5,6,7 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 19 --sparse_ratio 4.5 --kernels 'tanh' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-tanh-attention-diag-19-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-tanh-attention-diag-19-final

CUDA_VISIBLE_DEVICES=6,7 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 9 --sparse_ratio 0.5 --kernels 'tanh' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-tanh-attention-diag-9-spr-1w-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-tanh-attention-diag-9-spr-1w-final


CUDA_VISIBLE_DEVICES=4,5 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 19 --sparse_ratio 0.5 --kernels 'tanh' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-tanh-attention-diag-19-spr-1w-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-tanh-attention-diag-19-spr-1w-final





#######################################################
CUDA_VISIBLE_DEVICES=0,1 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 34 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name 'sparse-lowrank-tanh-attention-lowrank-only-check' --work_dir '/tanData/momentum_transformer/auto_lm/sparse-lowrank-tanh-attention-lowrank-only-check'




CUDA_VISIBLE_DEVICES=0,1 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 103 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --mu 0.9 --stepsize 0.9 --res_mu 0.2 --res_stepsize 1.0 --multi_gpu --use_wandb --project_name '2021-01--lm-2048-128' --job_name 'resmomentum-momentum-attention-mu-0-9-step-0-9-rmu-0-2-rstep-1-0-final-2' --work_dir '/tanData/momentum_transformer/auto_lm/resmomentum-momentum-attention-mu-0-9-step-0-9-rmu-0-2-rstep-1-0-final-2'

##########


CUDA_VISIBLE_DEVICES=0,1 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 1 --sparse_ratio 0.5 --kernels 'elu' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-elu-attention-diag-1-spr-1w-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-attention-diag-1-spr-1w-final

CUDA_VISIBLE_DEVICES=4,5 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 1 --sparse_ratio 2.5 --kernels 'elu' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-elu-attention-diag-1-spr-1sw-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-attention-diag-1-spr-1sw-final

CUDA_VISIBLE_DEVICES=6,7 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 19 --sparse_ratio 0.5 --kernels 'elu' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-elu-attention-diag-19-spr-1w-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-attention-diag-19-spr-1w-final

CUDA_VISIBLE_DEVICES=0,1 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 19 --sparse_ratio 2.5 --kernels 'elu' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-elu-attention-diag-19-spr-1sw-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-attention-diag-19-spr-1sw-final

CUDA_VISIBLE_DEVICES=2,3 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 19 --sparse_ratio 2.5 --kernels 'elu' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-elu-attention-diag-19-spr-1w-1-1-HD-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-attention-diag-19-spr-1w-1-1-HD-final


CUDA_VISIBLE_DEVICES=4,5 python ../src/train_lasso.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 19 --sparse_ratio 2.5 --kernels 'elu' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-elu-attention-diag-19-spr-1w-1-1-HD-sparse-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-attention-diag-19-spr-1w-1-1-HD-sparse-final



CUDA_VISIBLE_DEVICES=0,1 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 19 --sparse_ratio 2.5 --kernels 'elu' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-elu-attention-diag-19-spr-1w-1-1-HD-convex-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-attention-diag-19-spr-1w-1-1-HD-convex-final


CUDA_VISIBLE_DEVICES=2,3 python ../src/train_lasso.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 19 --sparse_ratio 2.5 --kernels 'elu' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-elu-attention-diag-19-spr-1w-1-1-HD-sparse-convex-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-attention-diag-19-spr-1w-1-1-HD-sparse-convex-final




CUDA_VISIBLE_DEVICES=2,3 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 29 --sparse_ratio 2.5 --kernels 'elu' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-elu-attention-diag-29-spr-1w-1-1-HD-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-attention-diag-29-spr-1w-1-1-HD-final


CUDA_VISIBLE_DEVICES=0,1 python ../src/train_lasso.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 29 --sparse_ratio 2.5 --kernels 'elu' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-elu-attention-diag-29-spr-1w-1-1-HD-sparse-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-attention-diag-29-spr-1w-1-1-HD-sparse-final


CUDA_VISIBLE_DEVICES=4,5 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 29 --sparse_ratio 4.5 --kernels 'elu' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-attention-diag-29-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-attention-diag-29-final



CUDA_VISIBLE_DEVICES=6,7 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 19 --sparse_ratio 2.5 --kernels 'elu' 'tanh' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-elu-tanh-attention-diag-19-spr-1w-1-1-HD-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-tanh-attention-diag-19-spr-1w-1-1-HD-final


CUDA_VISIBLE_DEVICES=0,1 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 19 --sparse_ratio 5.5 --kernels 'elu' 'tanh' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-elu-tanh-attention-lowrank-only-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-tanh-attention-lowrank-only-final


CUDA_VISIBLE_DEVICES=0,1 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 19 --sparse_ratio 2.5 --kernels 'elu' 'tanh' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-elu-tanh-attention-diag-19-spr-1w-1-1-HD-wonsparse-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-tanh-attention-diag-19-spr-1w-1-1-HD-wonsparse-final

CUDA_VISIBLE_DEVICES=2,3 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 19 --sparse_ratio 3.5 --kernels 'elu' 'tanh' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-elu-tanh-attention-diag-19-spr-1w-1-1-HD-wonboth-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-tanh-attention-diag-19-spr-1w-1-1-HD-wonboth-final

CUDA_VISIBLE_DEVICES=4,5 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 19 --sparse_ratio 2.5 --kernels 'tanh' 'tanh_orthogonal' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-tanh-tanh-orthogonal-attention-diag-19-spr-1w-1-1-HD-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-tanh-tanh-orthogonal-attention-diag-19-spr-1w-1-1-HD-final


CUDA_VISIBLE_DEVICES=2,3 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 1 --sparse_ratio 2.5 --kernels 'tanh' 'tanh_orthogonal' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-tanh-tanh-orthogonal-attention-diag-1-spr-1w-1-1-HD-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-tanh-tanh-orthogonal-attention-diag-1-spr-1w-1-1-HD-final


CUDA_VISIBLE_DEVICES=0,1 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 19 --sparse_ratio 5.5 --kernels 'tanh' 'tanh_orthogonal' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-tanh-tanh-orthogonal-attention-lowrank-only-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-tanh-tanh-orthogonal-attention-lowrank-only-final

CUDA_VISIBLE_DEVICES=0,1 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 19 --sparse_ratio 2.5 --kernels 'elu' 'sigmoid' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-elu-sigmoid-attention-diag-19-spr-1w-1-1-HD-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-sigmoid-attention-diag-19-spr-1w-1-1-HD-final

CUDA_VISIBLE_DEVICES=6,7 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 19 --sparse_ratio 2.5 --kernels 'elu' 'softplus' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-elu-softplus-attention-diag-19-spr-1w-1-1-HD-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-softplus-attention-diag-19-spr-1w-1-1-HD-final


CUDA_VISIBLE_DEVICES=4,5 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 19 --sparse_ratio 2.5 --kernels 'elu' 'elu_flip' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-elu-elu-flip-attention-diag-19-spr-1w-1-1-HD-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-elu-flip-attention-diag-19-spr-1w-1-1-HD-final


CUDA_VISIBLE_DEVICES=6,7 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 19 --sparse_ratio 5.5 --kernels 'elu' 'elu_flip' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-elu-elu-flip-attention-lowrank-only-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-elu-flip-attention-lowrank-only-final


CUDA_VISIBLE_DEVICES=2,3 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 19 --sparse_ratio 5.5 --kernels 'elu' 'sigmoid' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-elu-sigmoid-attention-lowrank-only-seed-0-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-sigmoid-attention-lowrank-only-seed-0-final --seed 0





CUDA_VISIBLE_DEVICES=2,3 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 20 --sparse_ratio 2.5 --kernels 'elu' 'elu_flip' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-elu-elu-flip-attention-diag-20-spr-1w-1-1-HD-sqk-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-elu-flip-attention-diag-20-spr-1w-1-1-HD-sqk-final



#HERE

# Terminator 4, GPU 2,3, a5
CUDA_VISIBLE_DEVICES=2,3 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 5 --sparse_ratio 2.5 --kernels 'elu' 'silu' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-elu-silu-attention-diag-5-spr-1w-1-1-HD-sqk-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-silu-attention-diag-5-spr-1w-1-1-HD-sqk-final


CUDA_VISIBLE_DEVICES=2,3 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 5 --sparse_ratio 2.5 --kernels 'elu' 'softplus' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-elu-softplus-attention-diag-5-spr-1w-1-1-HD-sqk-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-softplus-attention-diag-5-spr-1w-1-1-HD-sqk-final


CUDA_VISIBLE_DEVICES=2,3 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 5 --sparse_ratio 2.5 --kernels 'elu' 'sigmoid' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-elu-sigmoid-attention-diag-5-spr-1w-1-1-HD-sqk-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-sigmoid-attention-diag-5-spr-1w-1-1-HD-sqk-final

# Newton, GPU 6,7, a4
CUDA_VISIBLE_DEVICES=6,7 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 5 --sparse_ratio 4.5 --kernels 'elu' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-diag-5-sqk-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-diag-5-sqk-final

CUDA_VISIBLE_DEVICES=6,7 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 20 --sparse_ratio 4.5 --kernels 'elu' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-diag-20-sqk-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-diag-20-sqk-final

CUDA_VISIBLE_DEVICES=6,7 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 20 --sparse_ratio 2.5 --kernels 'elu' 'favor' --performer_proj_dim 4 --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-elu-flavor-4-attention-diag-20-spr-1w-1-1-HD-sqk-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-flavor-4-attention-diag-20-spr-1w-1-1-HD-sqk-final


CUDA_VISIBLE_DEVICES=6,7 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 20 --sparse_ratio 2.5 --kernels 'elu' 'elu_flip' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-elu-elu-flip-attention-diag-20-spr-1w-1-1-HD-wonlowrank-sqk-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-elu-flip-attention-diag-20-spr-1w-1-1-HD-wonlowrank-sqk-final

# Newton, GPU 0,1, a1
CUDA_VISIBLE_DEVICES=0,1 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 5 --sparse_ratio 2.5 --kernels 'elu' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-elu-attention-diag-5-spr-1w-1-1-HD-sqk-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-attention-diag-5-spr-1w-1-1-HD-sqk-final


CUDA_VISIBLE_DEVICES=0,1 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 20 --sparse_ratio 2.5 --kernels 'elu' 'elu_flip' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-elu-elu-flip-attention-diag-20-spr-1w-1-1-HD-sqk-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-elu-flip-attention-diag-20-spr-1w-1-1-HD-sqk-final


CUDA_VISIBLE_DEVICES=0,1 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 20 --sparse_ratio 5.5 --kernels 'elu' 'elu_flip' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-elu-elu-flip-attention-lowrank-only-check-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-elu-flip-attention-lowrank-only-check-final


# Newton, GPU 2,3, a2
CUDA_VISIBLE_DEVICES=2,3 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 5 --sparse_ratio 2.5 --kernels 'elu' 'elu_flip' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-elu-elu-flip-attention-diag-5-spr-1w-1-1-HD-sqk-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-elu-flip-attention-diag-5-spr-1w-1-1-HD-sqk-final

# Newton, GPU 4,5, a3
CUDA_VISIBLE_DEVICES=4,5 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 20 --sparse_ratio 2.5 --kernels 'elu' 'elu_flip' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-elu-elu-flip-attention-diag-20-spr-1w-1-1-HD-sqk-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-elu-flip-attention-diag-20-spr-1w-1-1-HD-sqk-final

CUDA_VISIBLE_DEVICES=4,5 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 20 --sparse_ratio 2.5 --kernels 'elu' 'favor' --performer_proj_dim 2 --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-elu-flavor-2-attention-diag-20-spr-1w-1-1-HD-wonlowrank-sqk-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-flavor-2-attention-diag-20-spr-1w-1-1-HD-wonlowrank-sqk-final


CUDA_VISIBLE_DEVICES=4,5 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 20 --sparse_ratio 4.5 --kernels 'elu' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-diag-20-sqk-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-diag-20-sqk-final

# Newton, GPU 0,1, a1
CUDA_VISIBLE_DEVICES=0,1 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 20 --sparse_ratio 2.5 --kernels 'elu' 'favor' --performer_proj_dim 8 --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-elu-flavor-attention-diag-20-spr-1w-1-1-HD-sqk-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-flavor-attention-diag-20-spr-1w-1-1-HD-sqk-final


CUDA_VISIBLE_DEVICES=0,1 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 20 --sparse_ratio 2.5 --kernels 'elu' 'favor' --performer_proj_dim 8 --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-elu-flavor-8-attention-diag-20-spr-1w-1-1-HD-wonlowrank-sqk-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-flavor-8-attention-diag-20-spr-1w-1-1-HD-wonlowrank-sqk-final

# Newton, GPU 2,3, a2
CUDA_VISIBLE_DEVICES=2,3 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 20 --sparse_ratio 2.5 --kernels 'elu' 'dpfp' --performer_proj_dim 8 --dpfp_n_roll 3 --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-elu-dpfp-3-attention-diag-20-spr-1w-1-1-HD-sqk-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-dpfp-3-attention-diag-20-spr-1w-1-1-HD-sqk-final

CUDA_VISIBLE_DEVICES=2,3 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 20 --sparse_ratio 2.5 --kernels 'elu' 'favor' --performer_proj_dim 2 --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-elu-flavor-2-attention-diag-20-spr-1w-1-1-HD-sqk-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-flavor-2-attention-diag-20-spr-1w-1-1-HD-sqk-final


CUDA_VISIBLE_DEVICES=2,3 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 20 --sparse_ratio 2.5 --kernels 'elu' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-elu-attention-diag-20-spr-1w-1-1-HD-sqk-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-attention-diag-20-spr-1w-1-1-HD-sqk-final


# Newton, GPU 4,5, a3
CUDA_VISIBLE_DEVICES=4,5 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 20 --sparse_ratio 2.5 --kernels 'elu' 'dpfp' --performer_proj_dim 8 --dpfp_n_roll 4 --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-elu-dpfp-4-attention-diag-20-spr-1w-1-1-HD-sqk-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-dpfp-4-attention-diag-20-spr-1w-1-1-HD-sqk-final

CUDA_VISIBLE_DEVICES=4,5 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 20 --sparse_ratio 2.5 --kernels 'elu' 'favor' --performer_proj_dim 1 --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-elu-flavor-1-attention-diag-20-spr-1w-1-1-HD-sqk-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-flavor-1-attention-diag-20-spr-1w-1-1-HD-sqk-final

CUDA_VISIBLE_DEVICES=0,1 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 20 --sparse_ratio 0.5 --kernels 'elu' 'favor' --performer_proj_dim 4 --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-elu-flavor-4-attention-diag-20-spr-0-5-sqk-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-flavor-4-attention-diag-20-spr-0-5-sqk-final


CUDA_VISIBLE_DEVICES=6,7 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 40 --sparse_ratio 2.5 --kernels 'elu' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-elu-attention-diag-40-spr-1w-1-1-HD-sqk-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-attention-diag-40-spr-1w-1-1-HD-sqk-final


# Newton, GPU 6,7, a4
CUDA_VISIBLE_DEVICES=6,7 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 20 --sparse_ratio 2.5 --kernels 'elu' 'favor' --performer_proj_dim 4 --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-elu-flavor-4-attention-diag-20-spr-1w-1-1-HD-sqk-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-flavor-4-attention-diag-20-spr-1w-1-1-HD-sqk-final


# Newton, GPU 6,7, a4
CUDA_VISIBLE_DEVICES=6,7 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 20 --sparse_ratio 5.5 --kernels 'elu' 'elu_flip' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-elu-elu-flip-attention-lowrank-only-check-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-elu-flip-attention-lowrank-only-check-final


CUDA_VISIBLE_DEVICES=6,7 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 5 --sparse_ratio 0.5 --kernels 'elu' 'favor' --performer_proj_dim 4 --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-elu-flavor-4-attention-diag-5-spr-0-5-sqk-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-flavor-4-attention-diag-5-spr-0-5-sqk-final


# Terminator, GPU 2,3,
CUDA_VISIBLE_DEVICES=2,3 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 20 --sparse_ratio 2.5 --kernels 'elu' 'favor' --performer_proj_dim 4 --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-elu-flavor-4-attention-diag-20-spr-1w-1-1-HD-wonlowrank-sqk-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-flavor-4-attention-diag-20-spr-1w-1-1-HD-wonlowrank-sqk-final


CUDA_VISIBLE_DEVICES=2,3 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 20 --sparse_ratio 5.5 --kernels 'elu' 'favor' --performer_proj_dim 4 --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-elu-favor-4-attention-lowrank-only-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-favor-4-attention-lowrank-only-final


CUDA_VISIBLE_DEVICES=2,3 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 20 --sparse_ratio 3.5 --kernels 'elu' 'elu_flip' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-elu-elu-flip-attention-diag-20-spr-1w-1-1-HD-wonboth-sqk-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-elu-flip-attention-diag-20-spr-1w-1-1-HD-wonboth-sqk-final


CUDA_VISIBLE_DEVICES=2,3 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 20 --sparse_ratio 2.5 --kernels 'elu' 'elu_flip' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-elu-elu-flip-attention-diag-20-wbetlowr-spr-1w-1-1-HD-sqk-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-elu-flip-attention-diag-20-wbetlowr-spr-1w-1-1-HD-sqk-final


CUDA_VISIBLE_DEVICES=4,5 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 20 --sparse_ratio 2.5 --kernels 'elu' 'elu_flip' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-elu-elu-flip-attention-diag-20-wconvexbetlowr-spr-1w-1-1-HD-sqk-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-elu-flip-attention-diag-20-wconvexbetlowr-spr-1w-1-1-HD-sqk-final





CUDA_VISIBLE_DEVICES=2,3 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 3 --sparse_ratio 2.5 --kernels 'elu' 'elu_flip' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-elu-elu-flip-attention-diag-3-spr-1w-1-1-HD-sqk-wonsparse-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-elu-flip-attention-diag-3-spr-1w-1-1-HD-sqk-wonsparse-final





CUDA_VISIBLE_DEVICES=0,1 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 20 --sparse_ratio 2.5 --kernels 'elu' 'elu_flip' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-elu-elu-flip-attention-diag-20-convex-spr-1w-1-1-HD-sqk-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-elu-flip-attention-diag-20-convex-spr-1w-1-1-HD-sqk-final


CUDA_VISIBLE_DEVICES=4,5 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 20 --sparse_ratio 0.5 --kernels 'elu' 'elu_flip' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-elu-elu-flip-attention-diag-20-scalar-wonlowrank-spr-1w-1-1-HD-sqk-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-elu-flip-attention-diag-20-scalar-wonlowrank-spr-1w-1-1-HD-sqk-final







CUDA_VISIBLE_DEVICES=2,3 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 2 --sparse_ratio 2.5 --kernels 'elu' 'elu_flip' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-elu-elu-flip-attention-diag-2-spr-1w-1-1-HD-sqk-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-elu-flip-attention-diag-2-spr-1w-1-1-HD-sqk-final


CUDA_VISIBLE_DEVICES=4,5 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 10 --sparse_ratio 2.5 --kernels 'elu' 'elu_flip' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-elu-elu-flip-attention-diag-10-spr-1w-1-1-HD-sqk-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-elu-flip-attention-diag-10-spr-1w-1-1-HD-sqk-final

CUDA_VISIBLE_DEVICES=0,1 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 5 --sparse_ratio 2.5 --kernels 'elu' 'elu_flip' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-elu-elu-flip-attention-diag-5-spr-1w-1-1-HD-sqk-wonsparse-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-elu-flip-attention-diag-5-spr-1w-1-1-HD-sqk-wonsparse-final

CUDA_VISIBLE_DEVICES=4,5 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 10 --sparse_ratio 2.5 --kernels 'elu' 'elu_flip' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-elu-elu-flip-attention-diag-10-spr-1w-1-1-HD-sqk-wonsparse-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-elu-flip-attention-diag-10-spr-1w-1-1-HD-sqk-wonsparse-final

CUDA_VISIBLE_DEVICES=2,3 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 3 --sparse_ratio 2.5 --kernels 'elu' 'elu_flip' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-elu-elu-flip-attention-diag-3-spr-1w-1-1-HD-sqk-wonsparse-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-elu-flip-attention-diag-3-spr-1w-1-1-HD-sqk-wonsparse-final


CUDA_VISIBLE_DEVICES=4,5 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 20 --sparse_ratio 2.5 --kernels 'elu' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-elu-attention-diag-20-spr-1w-1-1-HD-sqk-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-attention-diag-20-spr-1w-1-1-HD-sqk-final


CUDA_VISIBLE_DEVICES=4,5,6,7 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 20 --sparse_ratio 4.5 --kernels 'elu' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-diag-20-sqk-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-diag-20-sqk-final





CUDA_VISIBLE_DEVICES=0,1 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 20 --sparse_ratio 3.5 --kernels 'elu' 'elu_flip' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-elu-elu-flip-attention-diag-20-spr-1w-1-1-HD-sqk-wonboth-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-elu-flip-attention-diag-20-spr-1w-1-1-HD-sqk-wonboth-final

















CUDA_VISIBLE_DEVICES=2,3 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 1 --sparse_ratio 2.5 --kernels 'elu' 'elu_flip' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-elu-elu-flip-attention-diag-1-spr-1w-1-1-HD-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-elu-flip-attention-diag-1-spr-1w-1-1-HD-final







CUDA_VISIBLE_DEVICES=2,3 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 19 --sparse_ratio 2.5 --kernels 'elu' 'relu' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-elu-relu-attention-diag-19-spr-1w-1-1-HD-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-relu-attention-diag-19-spr-1w-1-1-HD-final










CUDA_VISIBLE_DEVICES=2,3 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 19 --sparse_ratio 2.5 --kernels 'elu' --multi_gpu --use_wandb --project_name 'debug' --job_name debug --work_dir /tanData/momentum_transformer/auto_lm/debug


CUDA_VISIBLE_DEVICES=2,3 python ../src/train_lasso.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 19 --sparse_ratio 2.5 --kernels 'elu' --multi_gpu --use_wandb --project_name 'debug' --job_name debug --work_dir /tanData/momentum_transformer/auto_lm/debug



CUDA_VISIBLE_DEVICES=2,3 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 19 --sparse_ratio 2.5 --kernels 'elu' --multi_gpu --use_wandb --project_name 'debug' --job_name sparse-lowrank-elu-attention-diag-19-spr-1cw-debug --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-attention-diag-19-spr-1cw-debug


CUDA_VISIBLE_DEVICES=0,1 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 1 --sparse_ratio 2.5 --kernels 'elu' 'elu_orthogonal' --multi_gpu --use_wandb --project_name 'debug' --job_name debug --work_dir /tanData/momentum_transformer/auto_lm/debug



CUDA_VISIBLE_DEVICES=2,3 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 4 --sparse_ratio 2.5 --kernels 'elu' 'elu_flip' --multi_gpu --use_wandb --project_name 'debug' --job_name debug --work_dir /tanData/momentum_transformer/auto_lm/debug




python /home/collab/tanmnguyen/repos/momentum-transformer/lmtool-fwms/src/train.py --cuda --data /home/collab/tanmnguyen/tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 256 --n_head 8 --d_head 32 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 400000 --attn_type 201 --tgt_len 384 --mem_len 0 --eval_tgt_len 384 --batch_size 56 --diag_size 20 --sparse_ratio 2.5 --kernels 'elu' 'elu_flip' --multi_gpu --project_name 'sparse-lowrank' --job_name sparse-lowrank-medium-elu-elu-flip-attention-diag-20-spr-1w-1-1-HD-sqk-final --work_dir /home/collab/tanmnguyen/results/momentum_transformer/auto_lm/sparse-lowrank-medium-elu-elu-flip-attention-diag-20-spr-1w-1-1-HD-sqk-final









####################################################
CUDA_VISIBLE_DEVICES=0,1 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 20 --sparse_ratio 0.5 --kernels 'elu' 'elu_flip' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-elu-elu-flip-attention-diag-20-ws-al1-bl2-spr-1w-1-1-HD-sqk-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-elu-flip-attention-diag-20-ws-al1-bl2-spr-1w-1-1-HD-sqk-final


CUDA_VISIBLE_DEVICES=6,7 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 20 --sparse_ratio 1.5 --kernels 'elu' 'elu_flip' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-elu-elu-flip-attention-diag-20-s-w-al1-bl2-spr-1w-1-1-HD-sqk-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-elu-flip-attention-diag-20-s-w-al1-bl2-spr-1w-1-1-HD-sqk-final

CUDA_VISIBLE_DEVICES=2,3 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 20 --sparse_ratio 3.5 --kernels 'elu' 'elu_flip' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-elu-elu-flip-attention-diag-20-ws-h-al1-bl2-spr-1w-1-1-HD-sqk-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-elu-flip-attention-diag-20-ws-h-al1-bl2-spr-1w-1-1-HD-sqk-final

CUDA_VISIBLE_DEVICES=4,5 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 20 --sparse_ratio 4.5 --kernels 'elu' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-diag-20-sqk-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-diag-20-sqk-final


######################################################
# Newton, GPU 0,1, a1
CUDA_VISIBLE_DEVICES=2,3 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 202 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 20 --sparse_ratio 2.5 --kernels 'elu' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-fastweight-elu-attention-diag-20-spr-a0-sqk-final --work_dir /tanData/momentum_transformer/auto_lm/fastweight-elu-attention-diag-20-spr-a0-sqk-final


CUDA_VISIBLE_DEVICES=4,5 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 202 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 20 --sparse_ratio 6.5 --kernels 'elu' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-fastweight-elu-attention-diag-20-spr-a0-b1-sqk-final --work_dir /tanData/momentum_transformer/auto_lm/fastweight-elu-attention-diag-20-spr-a0-b1-sqk-final

CUDA_VISIBLE_DEVICES=6,7 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 202 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 20 --sparse_ratio 7.5 --kernels 'elu' 'elu' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-fastweight-elu-elu-attention-diag-20-spr-a0a1-b1b0-sqk-final --work_dir /tanData/momentum_transformer/auto_lm/fastweight-elu-elu-attention-diag-20-spr-a0a1-b1b0-sqk-final

CUDA_VISIBLE_DEVICES=0,1 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 202 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 20 --sparse_ratio 8.5 --kernels 'elu' 'elu' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-fastweight-elu-elu-attention-diag-20-spr-a0-b1b0-sqk-final --work_dir /tanData/momentum_transformer/auto_lm/fastweight-elu-elu-attention-diag-20-spr-a0-b1b0-sqk-final


CUDA_VISIBLE_DEVICES=2,3 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 202 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 20 --sparse_ratio 9.5 --kernels 'elu' 'elu' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-fastweight-elu-elu-attention-a1-b0-sqk-final --work_dir /tanData/momentum_transformer/auto_lm/fastweight-elu-elu-attention-a1-b0-sqk-final








CUDA_VISIBLE_DEVICES=6,7 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 202 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 20 --sparse_ratio 2.5 --kernels 'dpfp' --dpfp_n_roll 1 --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-fastweight-dpfp-1-attention-diag-20-spr-a0-sqk-final --work_dir /tanData/momentum_transformer/auto_lm/fastweight-dpfp-1-attention-diag-20-spr-a0-sqk-final

CUDA_VISIBLE_DEVICES=0,1 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 202 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 20 --sparse_ratio 6.5 --kernels 'dpfp' --dpfp_n_roll 1 --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-fastweight-dpfp-1-attention-diag-20-spr-a0-b1-sqk-final --work_dir /tanData/momentum_transformer/auto_lm/fastweight-dpfp-1-attention-diag-20-spr-a0-b1-sqk-final



CUDA_VISIBLE_DEVICES=0,1 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 202 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 20 --sparse_ratio 6.5 --kernels 'elu' --multi_gpu --use_wandb --project_name 'debug' --job_name debug --work_dir /tanData/momentum_transformer/auto_lm/debug