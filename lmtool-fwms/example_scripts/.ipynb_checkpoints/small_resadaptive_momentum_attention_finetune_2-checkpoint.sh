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

export CUDA_VISIBLE_DEVICES=4,5,6,7
# requires 2 GPUs with 16 GB memory

echo 'Run training...'

python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 5000 --attn_type 102 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --mu 0.9 --stepsize 0.9 --res_stepsize 0.1 --res_delta 0.0001 --adaptive_type "wang" --multi_gpu --use_wandb --project_name '2021-01--lm-2048-128' --job_name 'resadaptive-momentum-attention-mu-0-9-step-0-9-rstep-0-1' --work_dir '/tanData/momentum_transformer/auto_lm/resadaptive-momentum-attention-mu-0-9-step-0-9-rstep-0-1'

python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 5000 --attn_type 102 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --mu 0.9 --stepsize 0.9 --res_stepsize 0.2 --res_delta 0.0001 --adaptive_type "wang" --multi_gpu --use_wandb --project_name '2021-01--lm-2048-128' --job_name 'resadaptive-momentum-attention-mu-0-9-step-0-9-rstep-0-2' --work_dir '/tanData/momentum_transformer/auto_lm/resadaptive-momentum-attention-mu-0-9-step-0-9-rstep-0-2'

python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 5000 --attn_type 102 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --mu 0.9 --stepsize 0.9 --res_stepsize 0.3 --res_delta 0.0001 --adaptive_type "wang" --multi_gpu --use_wandb --project_name '2021-01--lm-2048-128' --job_name 'resadaptive-momentum-attention-mu-0-9-step-0-9-rstep-0-3' --work_dir '/tanData/momentum_transformer/auto_lm/resadaptive-momentum-attention-mu-0-9-step-0-9-rstep-0-3'

python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 5000 --attn_type 102 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --mu 0.9 --stepsize 0.9 --res_stepsize 0.4 --res_delta 0.0001 --adaptive_type "wang" --multi_gpu --use_wandb --project_name '2021-01--lm-2048-128' --job_name 'resadaptive-momentum-attention-mu-0-9-step-0-9-rstep-0-4' --work_dir '/tanData/momentum_transformer/auto_lm/resadaptive-momentum-attention-mu-0-9-step-0-9-rstep-0-4'

python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 5000 --attn_type 102 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --mu 0.9 --stepsize 0.9 --res_stepsize 0.5 --res_delta 0.0001 --adaptive_type "wang" --multi_gpu --use_wandb --project_name '2021-01--lm-2048-128' --job_name 'resadaptive-momentum-attention-mu-0-9-step-0-9-rstep-0-5' --work_dir '/tanData/momentum_transformer/auto_lm/resadaptive-momentum-attention-mu-0-9-step-0-9-rstep-0-5'

python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 5000 --attn_type 102 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --mu 0.9 --stepsize 0.9 --res_stepsize 0.6 --res_delta 0.0001 --adaptive_type "wang" --multi_gpu --use_wandb --project_name '2021-01--lm-2048-128' --job_name 'resadaptive-momentum-attention-mu-0-9-step-0-9-rstep-0-6' --work_dir '/tanData/momentum_transformer/auto_lm/resadaptive-momentum-attention-mu-0-9-step-0-9-rstep-0-6'

python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 5000 --attn_type 102 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --mu 0.9 --stepsize 0.9 --res_stepsize 0.7 --res_delta 0.0001 --adaptive_type "wang" --multi_gpu --use_wandb --project_name '2021-01--lm-2048-128' --job_name 'resadaptive-momentum-attention-mu-0-9-step-0-9-rstep-0-7' --work_dir '/tanData/momentum_transformer/auto_lm/resadaptive-momentum-attention-mu-0-9-step-0-9-rstep-0-7'

python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 5000 --attn_type 102 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --mu 0.9 --stepsize 0.9 --res_stepsize 0.8 --res_delta 0.0001 --adaptive_type "wang" --multi_gpu --use_wandb --project_name '2021-01--lm-2048-128' --job_name 'resadaptive-momentum-attention-mu-0-9-step-0-9-rstep-0-8' --work_dir '/tanData/momentum_transformer/auto_lm/resadaptive-momentum-attention-mu-0-9-step-0-9-rstep-0-8'

python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 5000 --attn_type 102 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --mu 0.9 --stepsize 0.9 --res_stepsize 0.9 --res_delta 0.0001 --adaptive_type "wang" --multi_gpu --use_wandb --project_name '2021-01--lm-2048-128' --job_name 'resadaptive-momentum-attention-mu-0-9-step-0-9-rstep-0-9' --work_dir '/tanData/momentum_transformer/auto_lm/resadaptive-momentum-attention-mu-0-9-step-0-9-rstep-0-9'

python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 5000 --attn_type 102 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --mu 0.9 --stepsize 0.9 --res_stepsize 0.99 --res_delta 0.0001 --adaptive_type "wang" --multi_gpu --use_wandb --project_name '2021-01--lm-2048-128' --job_name 'resadaptive-momentum-attention-mu-0-9-step-0-9-rstep-0-99' --work_dir '/tanData/momentum_transformer/auto_lm/resadaptive-momentum-attention-mu-0-9-step-0-9-rstep-0-99'

python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 5000 --attn_type 102 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --mu 0.9 --stepsize 0.9 --res_stepsize 1.0 --res_delta 0.0001 --adaptive_type "wang" --multi_gpu --use_wandb --project_name '2021-01--lm-2048-128' --job_name 'resadaptive-momentum-attention-mu-0-9-step-0-9-rstep-1-0' --work_dir '/tanData/momentum_transformer/auto_lm/resadaptive-momentum-attention-mu-0-9-step-0-9-rstep-1-0'

python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 5000 --attn_type 102 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --mu 0.9 --stepsize 0.9 --res_stepsize 2.0 --res_delta 0.0001 --adaptive_type "wang" --multi_gpu --use_wandb --project_name '2021-01--lm-2048-128' --job_name 'resadaptive-momentum-attention-mu-0-9-step-0-9-rstep-2-0' --work_dir '/tanData/momentum_transformer/auto_lm/resadaptive-momentum-attention-mu-0-9-step-0-9-rstep-2-0'

python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 5000 --attn_type 102 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --mu 0.9 --stepsize 0.9 --res_stepsize 4.0 --res_delta 0.0001 --adaptive_type "wang" --multi_gpu --use_wandb --project_name '2021-01--lm-2048-128' --job_name 'resadaptive-momentum-attention-mu-0-9-step-0-9-rstep-4-0' --work_dir '/tanData/momentum_transformer/auto_lm/resadaptive-momentum-attention-mu-0-9-step-0-9-rstep-4-0'

python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 5000 --attn_type 102 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --mu 0.9 --stepsize 0.9 --res_stepsize 0.01 --res_delta 0.0001 --adaptive_type "wang" --multi_gpu --use_wandb --project_name '2021-01--lm-2048-128' --job_name 'resadaptive-momentum-attention-mu-0-9-step-0-9-rstep-0-01' --work_dir '/tanData/momentum_transformer/auto_lm/resadaptive-momentum-attention-mu-0-9-step-0-9-rstep-0-01'
