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

export CUDA_VISIBLE_DEVICES=2,3
# requires 2 GPUs with 16 GB memory

echo 'Run training...'

python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 5000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 0 --multi_gpu --use_wandb --project_name '2021-01--lm-2048-128' --job_name 'sparse-lowrank-attention-diag-0' --work_dir '/tanData/momentum_transformer/auto_lm/sparse-lowrank-attention-diag-0'

python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 5000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 1 --multi_gpu --use_wandb --project_name '2021-01--lm-2048-128' --job_name 'sparse-lowrank-attention-diag-1' --work_dir '/tanData/momentum_transformer/auto_lm/sparse-lowrank-attention-diag-1'

python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 5000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 2 --multi_gpu --use_wandb --project_name '2021-01--lm-2048-128' --job_name 'sparse-lowrank-attention-diag-2' --work_dir '/tanData/momentum_transformer/auto_lm/sparse-lowrank-attention-diag-2'

python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 5000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 3 --multi_gpu --use_wandb --project_name '2021-01--lm-2048-128' --job_name 'sparse-lowrank-attention-diag-3' --work_dir '/tanData/momentum_transformer/auto_lm/sparse-lowrank-attention-diag-3'

python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 5000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 4 --multi_gpu --use_wandb --project_name '2021-01--lm-2048-128' --job_name 'sparse-lowrank-attention-diag-4' --work_dir '/tanData/momentum_transformer/auto_lm/sparse-lowrank-attention-diag-4'

python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 5000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 5 --multi_gpu --use_wandb --project_name '2021-01--lm-2048-128' --job_name 'sparse-lowrank-attention-diag-5' --work_dir '/tanData/momentum_transformer/auto_lm/sparse-lowrank-attention-diag-5'

python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 5000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 6 --multi_gpu --use_wandb --project_name '2021-01--lm-2048-128' --job_name 'sparse-lowrank-attention-diag-6' --work_dir '/tanData/momentum_transformer/auto_lm/sparse-lowrank-attention-diag-6'

python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 5000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 7 --multi_gpu --use_wandb --project_name '2021-01--lm-2048-128' --job_name 'sparse-lowrank-attention-diag-7' --work_dir '/tanData/momentum_transformer/auto_lm/sparse-lowrank-attention-diag-7'

python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 5000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 8 --multi_gpu --use_wandb --project_name '2021-01--lm-2048-128' --job_name 'sparse-lowrank-attention-diag-8' --work_dir '/tanData/momentum_transformer/auto_lm/sparse-lowrank-attention-diag-8'


