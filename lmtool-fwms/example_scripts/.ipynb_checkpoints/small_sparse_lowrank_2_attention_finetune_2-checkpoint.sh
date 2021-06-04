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

export CUDA_VISIBLE_DEVICES=6,7
# requires 2 GPUs with 16 GB memory

echo 'Run training...'

python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 5000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 9 --sparse_ratio 0.5 --multi_gpu --use_wandb --project_name '2021-01--lm-2048-128' --job_name 'sparse-lowrank-attention-diag-9-spr-0-5' --work_dir '/tanData/momentum_transformer/auto_lm/sparse-lowrank-attention-diag-9-spr-0-5'

python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 5000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 19 --sparse_ratio 0.5 --multi_gpu --use_wandb --project_name '2021-01--lm-2048-128' --job_name 'sparse-lowrank-attention-diag-19-spr-0-5' --work_dir '/tanData/momentum_transformer/auto_lm/sparse-lowrank-attention-diag-19-spr-0-5'

python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 5000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 29 --sparse_ratio 0.5 --multi_gpu --use_wandb --project_name '2021-01--lm-2048-128' --job_name 'sparse-lowrank-attention-diag-29-spr-0-5' --work_dir '/tanData/momentum_transformer/auto_lm/sparse-lowrank-attention-diag-29-spr-0-5'

python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 5000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 49 --sparse_ratio 0.5 --multi_gpu --use_wandb --project_name '2021-01--lm-2048-128' --job_name 'sparse-lowrank-attention-diag-49-spr-0-5' --work_dir '/tanData/momentum_transformer/auto_lm/sparse-lowrank-attention-diag-49-spr-0-5'

python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 5000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 99 --sparse_ratio 0.5 --multi_gpu --use_wandb --project_name '2021-01--lm-2048-128' --job_name 'sparse-lowrank-attention-diag-99-spr-0-5' --work_dir '/tanData/momentum_transformer/auto_lm/sparse-lowrank-attention-diag-99-spr-0-5'

python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 5000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 149 --sparse_ratio 0.5 --multi_gpu --use_wandb --project_name '2021-01--lm-2048-128' --job_name 'sparse-lowrank-attention-diag-149-spr-0-5' --work_dir '/tanData/momentum_transformer/auto_lm/sparse-lowrank-attention-diag-149-spr-0-5'

python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 5000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 199 --sparse_ratio 0.5 --multi_gpu --use_wandb --project_name '2021-01--lm-2048-128' --job_name 'sparse-lowrank-attention-diag-199-spr-0-5' --work_dir '/tanData/momentum_transformer/auto_lm/sparse-lowrank-attention-diag-199-spr-0-5'

python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 5000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 255 --sparse_ratio 0.5 --multi_gpu --use_wandb --project_name '2021-01--lm-2048-128' --job_name 'sparse-lowrank-attention-diag-255-spr-0-5' --work_dir '/tanData/momentum_transformer/auto_lm/sparse-lowrank-attention-diag-255-spr-0-5'

python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 34 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --multi_gpu --use_wandb --project_name '2021-01--lm-2048-128' --job_name 'linear-attention-sanity-check' --work_dir '/tanData/momentum_transformer/auto_lm/linear-attention-sanity-check'

python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 5000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 255 --sparse_ratio 1.0 --multi_gpu --use_wandb --project_name '2021-01--lm-2048-128' --job_name 'sparse-lowrank-attention-diag-255-spr-1-0' --work_dir '/tanData/momentum_transformer/auto_lm/sparse-lowrank-attention-diag-255-spr-1-0'
