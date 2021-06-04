#!/bin/bash

CUDA_VISIBLE_DEVICES=4,5 python ../src/eval_sliding_window.py --cuda --batch_size 1 --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --tgt_len 256 --mem_len 0 --clamp_len 256 --split valid --work_dir '/tanData/momentum_transformer/auto_lm/softmax-attention-final'

CUDA_VISIBLE_DEVICES=4,5 python ../src/eval_sliding_window.py --cuda --batch_size 1 --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --tgt_len 256 --mem_len 0 --clamp_len 256 --split test --work_dir '/tanData/momentum_transformer/auto_lm/softmax-attention-final'

CUDA_VISIBLE_DEVICES=4,5 python ../src/eval_sliding_window.py --cuda --batch_size 1 --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --tgt_len 256 --mem_len 0 --clamp_len 256 --split valid --work_dir '/tanData/momentum_transformer/auto_lm/resadaptive-momentum-attention-mu-0-9-step-0-9-rstep-1-0-final'

CUDA_VISIBLE_DEVICES=4,5 python ../src/eval_sliding_window.py --cuda --batch_size 1 --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --tgt_len 256 --mem_len 0 --clamp_len 256 --split test --work_dir '/tanData/momentum_transformer/auto_lm/resadaptive-momentum-attention-mu-0-9-step-0-9-rstep-1-0-final'


