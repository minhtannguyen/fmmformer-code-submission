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

export CUDA_VISIBLE_DEVICES=2,3
# requires 2 GPUs with 16 GB memory

declare -a StringArray=("tanh")
declare -a IntegerArray=(0 1 2 3 4 5)

# declare -a StringArray=("elu" "tanh" "relu" "celu" "sigmoid" "leaky_relu" "softplus")

for kname in ${StringArray[@]}
do
    for dsize in ${IntegerArray[@]}
    do
        python ../src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 5000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --diag_size ${dsize} --sparse_ratio 1.5 --kernels ${kname} --multi_gpu --use_wandb --project_name '2021-01--lm-2048-128' --job_name sparse-lowrank-${kname}-attention-diag-${dsize}-spr-2w --work_dir /tanData/momentum_transformer/auto_lm/sparse-lowrank-${kname}-attention-diag-${dsize}-spr-2w
    done
done




