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

# requires 2 GPUs with 16 GB memory

CUDA_VISIBLE_DEVICES=4,5 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 101 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --mu 0.9 --stepsize 0.9 --multi_gpu --use_wandb --project_name '2021-01--lm-2048-128' --job_name 'momentum-attention-mu-0-9-step-0-9-final' --work_dir '/tanData/momentum_transformer/auto_lm/momentum-attention-mu-0-9-step-0-9-final'


CUDA_VISIBLE_DEVICES=2,3 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 102 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --mu 0.9 --stepsize 0.9 --res_stepsize 0.9 --res_delta 0.0001 --adaptive_type "wang" --multi_gpu --use_wandb --project_name '2021-01--lm-2048-128' --job_name 'resadaptive-momentum-attention-mu-0-9-step-0-9-rstep-0-9' --work_dir '/tanData/momentum_transformer/auto_lm/resadaptive-momentum-attention-mu-0-9-step-0-9-rstep-0-9'

CUDA_VISIBLE_DEVICES=4,5,6,7 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 5000 --attn_type 103 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --mu 0.9 --stepsize 0.9 --res_mu 0.8 --res_stepsize 0.5 --multi_gpu --use_wandb --project_name 'debug' --job_name 'resmomentum-momentum-attention-mu-0-9-step-0-9-rstep-0-9' --work_dir '/tanData/momentum_transformer/auto_lm/resmomentum-momentum-attention-mu-0-9-step-0-9-rstep-0-9'


CUDA_VISIBLE_DEVICES=6,7 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 34 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --multi_gpu --use_wandb --project_name '2021-01--lm-2048-128' --job_name 'linear-attention-final' --work_dir '/tanData/momentum_transformer/auto_lm/linear-attention-final'

CUDA_VISIBLE_DEVICES=0,1 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 2 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --multi_gpu --use_wandb --project_name '2021-01--lm-2048-128' --job_name 'softmax-attention-final' --work_dir '/tanData/momentum_transformer/auto_lm/softmax-attention-final'

CUDA_VISIBLE_DEVICES=2,3 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 102 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --mu 0.9 --stepsize 0.9 --res_stepsize 1.0 --res_delta 0.0001 --adaptive_type "wang" --multi_gpu --use_wandb --project_name '2021-01--lm-2048-128' --job_name 'resadaptive-momentum-attention-mu-0-9-step-0-9-rstep-1-0-final' --work_dir '/tanData/momentum_transformer/auto_lm/resadaptive-momentum-attention-mu-0-9-step-0-9-rstep-1-0-final'

CUDA_VISIBLE_DEVICES=4,5 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 103 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --mu 0.9 --stepsize 0.9 --res_mu 0.2 --res_stepsize 1.0 --multi_gpu --use_wandb --project_name '2021-01--lm-2048-128' --job_name 'resmomentum-momentum-attention-mu-0-9-step-0-9-rmu-0-2-rstep-1-0-final' --work_dir '/tanData/momentum_transformer/auto_lm/resmomentum-momentum-attention-mu-0-9-step-0-9-rmu-0-2-rstep-1-0-final'

CUDA_VISIBLE_DEVICES=2,3 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 255 --multi_gpu --use_wandb --project_name 'debug' --job_name 'sparse-lowrank-attention-debug' --work_dir '/tanData/momentum_transformer/auto_lm/sparse-lowrank-attention-debug'


CUDA_VISIBLE_DEVICES=2,3 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 2 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --multi_gpu --use_wandb --project_name 'debug' --job_name 'softmax-attention-debug' --work_dir '/tanData/momentum_transformer/auto_lm/softmax-attention-debug'


CUDA_VISIBLE_DEVICES=2,3 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 34 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --multi_gpu --use_wandb --project_name 'debug' --job_name 'linear-attention-debug' --work_dir '/tanData/momentum_transformer/auto_lm/linear-attention-debug'


############

CUDA_VISIBLE_DEVICES=0,1 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 5000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 255 --multi_gpu --use_wandb --project_name 'debug' --job_name 'sparse-lowrank-attention-run-1111' --work_dir '/tanData/momentum_transformer/auto_lm/sparse-lowrank-attention-run-1111'


CUDA_VISIBLE_DEVICES=2,3 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 5000 --attn_type 2 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --multi_gpu --use_wandb --project_name 'debug' --job_name 'softmax-attention-run-1111' --work_dir '/tanData/momentum_transformer/auto_lm/softmax-attention-run-1111'

############

CUDA_VISIBLE_DEVICES=0,1 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 5000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 255 --sparse_ratio 0.8 --multi_gpu --use_wandb --project_name 'debug' --job_name 'sparse-lowrank-attention-debug' --work_dir '/tanData/momentum_transformer/auto_lm/sparse-lowrank-attention-debug'


CUDA_VISIBLE_DEVICES=0,1 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 5000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 255 --sparse_ratio 0.8 --kernels 'elu' 'tanh' --multi_gpu --use_wandb --project_name 'debug' --job_name 'sparse-lowrank-attention-debug' --work_dir '/tanData/momentum_transformer/auto_lm/sparse-lowrank-attention-debug'

CUDA_VISIBLE_DEVICES=0,1 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 5000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 255 --sparse_ratio 0.5 --kernels 'elu' --multi_gpu --use_wandb --project_name 'debug' --job_name 'sparse-lowrank-attention-debug-2' --work_dir '/tanData/momentum_transformer/auto_lm/sparse-lowrank-attention-debug-2'







CUDA_VISIBLE_DEVICES=4,5 python ../src/eval_sliding_window.py --cuda --batch_size 1 --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --tgt_len 256 --mem_len 0 --clamp_len 256 --split valid --work_dir '/tanData/momentum_transformer/auto_lm/linear-attention-final'

CUDA_VISIBLE_DEVICES=6,7 python ../src/eval_sliding_window.py --cuda --batch_size 1 --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --tgt_len 256 --mem_len 0 --clamp_len 256 --split test --work_dir '/tanData/momentum_transformer/auto_lm/linear-attention-final'

CUDA_VISIBLE_DEVICES=4,5 python ../src/eval_sliding_window.py --cuda --batch_size 1 --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --tgt_len 256 --mem_len 0 --clamp_len 256 --split valid --work_dir '/tanData/momentum_transformer/auto_lm/sparse-lowrank-tanh-attention-diag-1-spr-1w-getw'

CUDA_VISIBLE_DEVICES=0,1 python ../src/eval_sliding_window.py --cuda --batch_size 1 --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --tgt_len 256 --mem_len 0 --clamp_len 256 --split test --work_dir '/tanData/momentum_transformer/auto_lm/sparse-lowrank-tanh-attention-diag-1-spr-1w-getw'


bash small_linear_attention.sh valid --work_dir '/tanData/momentum_transformer/auto_lm/linear-attention-final'

--attn_type 201 --diag_size 1 --sparse_ratio 0.5 --kernels 'tanh'


CUDA_VISIBLE_DEVICES=0,1 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 5000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 255 --sparse_ratio 0.5 --kernels 'elu' --skip_attn_normalization --multi_gpu --use_wandb --project_name 'debug' --job_name 'sparse-lowrank-attention-debug-2' --work_dir '/tanData/momentum_transformer/auto_lm/sparse-lowrank-attention-debug-2'


CUDA_VISIBLE_DEVICES=0,1 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 5000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 255 --sparse_ratio 1.5 --kernels 'elu' --skip_attn_normalization --multi_gpu --use_wandb --project_name 'debug' --job_name 'sparse-lowrank-attention-debug-2' --work_dir '/tanData/momentum_transformer/auto_lm/sparse-lowrank-attention-debug-2'


CUDA_VISIBLE_DEVICES=0,1 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 5000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 255 --sparse_ratio 4.5 --kernels 'elu' --skip_attn_normalization --multi_gpu --use_wandb --project_name 'debug' --job_name 'sparse-lowrank-attention-debug-2' --work_dir '/tanData/momentum_transformer/auto_lm/sparse-lowrank-attention-debug-2'


CUDA_VISIBLE_DEVICES=0,1 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 34 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --multi_gpu --use_wandb --project_name 'debug' --job_name 'linear-attention-debug' --work_dir '/tanData/momentum_transformer/auto_lm/linear-attention-debug'



python /home/collab/tanmnguyen/repos/momentum-transformer/lmtool-fwms/src/train.py --cuda --data /home/collab/tanmnguyen/tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 101 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --mu 0.9 --stepsize 0.9 --multi_gpu --use_wandb --project_name '2021-01--lm-2048-128' --job_name momentum-attention-mu-0-9-step-0-9-final --work_dir /home/collab/tanmnguyen/results/momentum_transformer/auto_lm/momentum-attention-mu-0-9-step-0-9-final >/home/collab/tanmnguyen/results/debug.txt



python /home/collab/tanmnguyen/repos/momentum-transformer/lmtool-fwms/src/train.py --cuda --data /home/collab/tanmnguyen/tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 19 --sparse_ratio 4.5 --kernels 'tanh' --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-tanh-attention-diag-19-medusa-final --work_dir /home/collab/tanmnguyen/results/momentum_transformer/auto_lm/sparse-lowrank-tanh-attention-diag-19-medusa-final >/home/collab/tanmnguyen/results/debug.txt


python /home/collab/tanmnguyen/repos/momentum-transformer/lmtool-fwms/src/train.py --cuda --data /home/collab/tanmnguyen/tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 48000 --attn_type 102 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --mu 0.9 --stepsize 0.9 --res_stepsize 1.0 --res_delta 0.0001 --adaptive_type "wang" --multi_gpu --project_name '2021-01--lm-2048-128' --job_name resadaptive-momentum-attention-mu-0-9-step-0-9-rstep-1-0-finetune --work_dir /home/collab/tanmnguyen/results/momentum_transformer/auto_lm/resadaptive-momentum-attention-mu-0-9-step-0-9-rstep-1-0-finetune



CUDA_VISIBLE_DEVICES=0 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 2 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 1000 --project_name '2021-01--lm-2048-128' --job_name 'softmax-attention-final-test' --work_dir '/tanData/momentum_transformer/auto_lm/softmax-attention-final-test' --restart --restart_dir '/tanData/momentum_transformer/auto_lm/softmax-attention-final'


CUDA_VISIBLE_DEVICES=0 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 2 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 1000 --project_name 'debug' --job_name 'sparse-lowrank-elu-elu-flip-attention-diag-20-spr-1w-1-1-HD-sqk-test' --work_dir '/tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-elu-flip-attention-diag-20-spr-1w-1-1-HD-sqk-test-temp' --restart --restart_dir '/tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-elu-flip-attention-diag-20-spr-1w-1-1-HD-sqk-test'


CUDA_VISIBLE_DEVICES=6,7 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 20 --sparse_ratio 2.5 --kernels 'elu' 'elu_flip' --multi_gpu --project_name 'debug' --job_name 'sparse-lowrank-elu-elu-flip-attention-diag-20-spr-1w-1-1-HD-sqk-test' --work_dir '/tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-elu-flip-attention-diag-20-spr-1w-1-1-HD-sqk-test-temp' --restart --restart_dir '/tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-elu-flip-attention-diag-20-spr-1w-1-1-HD-sqk-test'



CUDA_VISIBLE_DEVICES=4,5 python ../src/eval.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --split 'test' --batch_size 96 --tgt_len 256 --mem_len 0 --clamp_len 256 --work_dir '/tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-attention-diag-40-spr-1w-1-1-HD-sqk-final'



CUDA_VISIBLE_DEVICES=4,5 python ../src/eval_sliding_window.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --split 'test' --batch_size 96 --tgt_len 256 --mem_len 0 --clamp_len 256 --work_dir '/tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-attention-diag-40-spr-1w-1-1-HD-sqk-final'

CUDA_VISIBLE_DEVICES=6,7 python ../src/eval.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --split 'test' --batch_size 96 --tgt_len 256 --mem_len 0 --clamp_len 256 --work_dir '/tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-flavor-4-attention-diag-20-spr-0-5-sqk-final'



CUDA_VISIBLE_DEVICES=2,3 python ../src/eval.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --split 'test' --batch_size 96 --tgt_len 256 --mem_len 0 --clamp_len 256 --work_dir '/tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-elu-flip-attention-diag-20-spr-1w-1-1-HD-sqk-final'

CUDA_VISIBLE_DEVICES=2 python ../src/eval_sliding_window.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --split 'valid' --batch_size 1 --tgt_len 256 --mem_len 0 --clamp_len 256 --work_dir '/tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-attention-diag-20-spr-1w-1-1-HD-sqk-final'

CUDA_VISIBLE_DEVICES=3 python ../src/eval_sliding_window.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --split 'test' --batch_size 1 --tgt_len 256 --mem_len 0 --clamp_len 256 --work_dir '/tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-attention-diag-20-spr-1w-1-1-HD-sqk-final'



CUDA_VISIBLE_DEVICES=0 python ../src/eval_sliding_window.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --split 'valid' --batch_size 96 --tgt_len 256 --mem_len 0 --clamp_len 256 --work_dir '/tanData/momentum_transformer/auto_lm/fastweight-elu-elu-attention-diag-20-spr-a0a1-b1b0-sqk-test'


CUDA_VISIBLE_DEVICES=0 python ../src/eval_sliding_window.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --split 'valid' --batch_size 96 --tgt_len 256 --mem_len 0 --clamp_len 256 --work_dir '/tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-attention-diag-20-spr-1w-1-1-HD-sqk-final'






--adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 20 --sparse_ratio 2.5 --kernels 'elu' 'elu_flip' --multi_gpu --project_name 'debug' --job_name 'sparse-lowrank-elu-elu-flip-attention-diag-20-spr-1w-1-1-HD-sqk-test' --work_dir '/tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-elu-flip-attention-diag-20-spr-1w-1-1-HD-sqk-test-temp' --restart --restart_dir '/tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-elu-flip-attention-diag-20-spr-1w-1-1-HD-sqk-test'



# Newton, GPU 4,5, a3
CUDA_VISIBLE_DEVICES=2,3 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 20 --sparse_ratio 2.5 --kernels 'elu' 'favor' --performer_proj_dim 8 --multi_gpu --use_wandb --project_name 'debug' --job_name debug --work_dir /tanData/momentum_transformer/auto_lm/debug


CUDA_VISIBLE_DEVICES=2,3 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 20 --sparse_ratio 2.5 --kernels 'elu' 'dpfp' --performer_proj_dim 8 --dpfp_n_roll 1 --multi_gpu --use_wandb --project_name 'debug' --job_name debug --work_dir /tanData/momentum_transformer/auto_lm/debug



CUDA_VISIBLE_DEVICES=2,3 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 20 --sparse_ratio 2.5 --kernels 'elu' 'elu_flip' --multi_gpu --use_wandb --project_name 'debug' --job_name debug --work_dir /tanData/momentum_transformer/auto_lm/debug




CUDA_VISIBLE_DEVICES=0,1 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 24 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name 'fastweigt-elu-attention-final' --work_dir '/tanData/momentum_transformer/auto_lm/fastweigt-elu-attention-final'


CUDA_VISIBLE_DEVICES=2,3 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 26 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --dpfp_n_roll 1 --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name 'fastweigt-dpfp-1-attention-final' --work_dir '/tanData/momentum_transformer/auto_lm/fastweigt-dpfp-1-attention-final'


CUDA_VISIBLE_DEVICES=4,5 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 26 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --dpfp_n_roll 2 --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name 'fastweigt-dpfp-2-attention-final' --work_dir '/tanData/momentum_transformer/auto_lm/fastweigt-dpfp-2-attention-final'



CUDA_VISIBLE_DEVICES=4,5 python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size 20 --sparse_ratio 2.5 --kernels 'elu' 'dpfp' --performer_proj_dim 8 --dpfp_n_roll 4 --multi_gpu --use_wandb --project_name 'sparse-lowrank' --job_name sparse-lowrank-elu-dpfp-4-attention-diag-20-spr-1w-1-1-HD-sqk-final --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-dpfp-4-attention-diag-20-spr-1w-1-1-HD-sqk-final