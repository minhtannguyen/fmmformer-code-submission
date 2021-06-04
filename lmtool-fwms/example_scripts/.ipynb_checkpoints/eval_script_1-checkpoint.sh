#!/bin/bash

# CUDA_VISIBLE_DEVICES=4,5 python ../src/eval_sliding_window.py --cuda --batch_size 1 --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --tgt_len 256 --mem_len 0 --clamp_len 256 --split valid --work_dir '/tanData/momentum_transformer/auto_lm/softmax-attention-final'

# CUDA_VISIBLE_DEVICES=4,5 python ../src/eval_sliding_window.py --cuda --batch_size 1 --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --tgt_len 256 --mem_len 0 --clamp_len 256 --split test --work_dir '/tanData/momentum_transformer/auto_lm/softmax-attention-final'


CUDA_VISIBLE_DEVICES=0,1 python ../src/eval_sliding_window.py --cuda --batch_size 1 --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --tgt_len 256 --mem_len 0 --clamp_len 256 --split valid --work_dir '/tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-attention-diag-5-spr-1w-1-1-HD-sqk-final'

CUDA_VISIBLE_DEVICES=0,1 python ../src/eval_sliding_window.py --cuda --batch_size 1 --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --tgt_len 256 --mem_len 0 --clamp_len 256 --split test --work_dir '/tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-attention-diag-5-spr-1w-1-1-HD-sqk-final'

CUDA_VISIBLE_DEVICES=0,1 python ../src/eval_sliding_window.py --cuda --batch_size 1 --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --tgt_len 256 --mem_len 0 --clamp_len 256 --split valid --work_dir '/tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-elu-flip-attention-lowrank-only-final'

CUDA_VISIBLE_DEVICES=0,1 python ../src/eval_sliding_window.py --cuda --batch_size 1 --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --tgt_len 256 --mem_len 0 --clamp_len 256 --split test --work_dir '/tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-elu-flip-attention-lowrank-only-final'




CUDA_VISIBLE_DEVICES=2,3 python ../src/eval_sliding_window.py --cuda --batch_size 1 --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --tgt_len 256 --mem_len 0 --clamp_len 256 --split valid --work_dir '/tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-elu-flip-attention-diag-5-spr-1w-1-1-HD-sqk-final'

CUDA_VISIBLE_DEVICES=2,3 python ../src/eval_sliding_window.py --cuda --batch_size 1 --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --tgt_len 256 --mem_len 0 --clamp_len 256 --split test --work_dir '/tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-elu-flip-attention-diag-5-spr-1w-1-1-HD-sqk-final'


CUDA_VISIBLE_DEVICES=4,5 python ../src/eval_sliding_window.py --cuda --batch_size 1 --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --tgt_len 256 --mem_len 0 --clamp_len 256 --split valid --work_dir '/tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-elu-flip-attention-diag-20-spr-1w-1-1-HD-sqk-final'

CUDA_VISIBLE_DEVICES=4,5 python ../src/eval_sliding_window.py --cuda --batch_size 1 --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --tgt_len 256 --mem_len 0 --clamp_len 256 --split test --work_dir '/tanData/momentum_transformer/auto_lm/sparse-lowrank-elu-elu-flip-attention-diag-20-spr-1w-1-1-HD-sqk-final'


CUDA_VISIBLE_DEVICES=6,7 python ../src/eval_sliding_window.py --cuda --batch_size 1 --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --tgt_len 256 --mem_len 0 --clamp_len 256 --split valid --work_dir '/tanData/momentum_transformer/auto_lm/sparse-lowrank-diag-5-sqk-final'

CUDA_VISIBLE_DEVICES=6,7 python ../src/eval_sliding_window.py --cuda --batch_size 1 --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --tgt_len 256 --mem_len 0 --clamp_len 256 --split test --work_dir '/tanData/momentum_transformer/auto_lm/sparse-lowrank-diag-5-sqk-final'


CUDA_VISIBLE_DEVICES=7 python ../src/eval_sliding_window.py --cuda --batch_size 1 --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --tgt_len 256 --mem_len 0 --clamp_len 256 --split test --work_dir '/tanData/momentum_transformer/auto_lm/momentum-attention-mu-0-9-step-0-9-medusa-final'

CUDA_VISIBLE_DEVICES=7 python ../src/eval_sliding_window.py --cuda --batch_size 1 --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --tgt_len 256 --mem_len 0 --clamp_len 256 --split valid --work_dir '/tanData/momentum_transformer/auto_lm/momentum-attention-mu-0-9-step-0-9-medusa-final'