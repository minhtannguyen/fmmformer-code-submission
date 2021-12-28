#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 2 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name softmax-seed-1111 --work_dir /tanData/mgattn/softmax-seed-1111


CUDA_VISIBLE_DEVICES=4,5 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 2 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name softmax-seed-1111 --work_dir /tanData/mgattn/debug

CUDA_VISIBLE_DEVICES=2,3 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 200 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --update_mode 'soft' --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name mgk-mulgauss-soft-e-n-head-seed-1111 --work_dir /tanData/mgattn/mgk-mulgauss-soft-e-n-head-seed-1111


CUDA_VISIBLE_DEVICES=2,3 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 200 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name mgk-mulgauss-hard-e-seed-1111 --work_dir /tanData/mgattn/mgk-mulgauss-hard-e-seed-1111


CUDA_VISIBLE_DEVICES=4,5 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 200 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --update_mode 'soft' --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name mgk-mulgauss-soft-e-seed-1111 --work_dir /tanData/mgattn/mgk-mulgauss-soft-e-seed-1111


CUDA_VISIBLE_DEVICES=6,7 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 200 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --update_mode 'rbf' --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name mgk-mulgauss-rbf-pi-reg-0-nhead-klen-seed-1111 --work_dir /tanData/mgattn/mgk-mulgauss-rbf-pi-reg-0-nhead-klen-seed-1111 --pi_reg 0.0

CUDA_VISIBLE_DEVICES=4,5 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 200 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --update_mode 'soft' --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name mgk-mulgauss-soft-e-nhead-klen-seed-1111 --work_dir /tanData/mgattn/mgk-mulgauss-soft-e-nhead-klen-seed-1111

CUDA_VISIBLE_DEVICES=2,3 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 4 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 200 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --update_mode 'rbf' --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name mgk-mulgauss-rbf-pi-reg-0-nhead-klen-4head-seed-1111 --work_dir /tanData/mgattn/mgk-mulgauss-rbf-pi-reg-0-nhead-klen-4head-seed-1111 --pi_reg 0.0


CUDA_VISIBLE_DEVICES=2,3 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 200 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --update_mode 'rbf2keys' --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name mgk-2keys-rbf-pi-reg-0-nhead-klen-8head-seed-1111 --work_dir /tanData/mgattn/mgk-2keys-rbf-pi-reg-0-nhead-klen-8head-seed-1111 --pi_reg 0.0




CUDA_VISIBLE_DEVICES=6,7 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 4 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 200 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --update_mode 'rbf2keys' --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name mgk-2keys-rbf-pi-reg-0-nhead-klen-4head-diffbwk2small4x-seed-1111 --work_dir /tanData/mgattn/mgk-2keys-rbf-pi-reg-0-nhead-klen-4head-diffbwk2small4x-seed-1111 --pi_reg 0.0

CUDA_VISIBLE_DEVICES=2,3 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 4 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 200 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --update_mode 'rbf2keys' --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name mgk-2keys-rbf-pi-reg-0-nhead-klen-4head-diffbwk2small8x-seed-1111 --work_dir /tanData/mgattn/mgk-2keys-rbf-pi-reg-0-nhead-klen-4head-diffbwk2small8x-seed-1111 --pi_reg 0.0


CUDA_VISIBLE_DEVICES=2,3 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 4 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 200 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --update_mode 'rbf2keys' --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name mgk-2keys-rbf-pi-reg-0-nhead-klen-4head-diffbwk2small6x-seed-1111 --work_dir /tanData/mgattn/mgk-2keys-rbf-pi-reg-0-nhead-klen-4head-diffbwk2small6x-seed-1111 --pi_reg 0.0


CUDA_VISIBLE_DEVICES=4,5 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 4 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 200 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --update_mode 'rbf2keys' --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name mgk-2keys-rbf-pi-reg-0-nhead-klen-4head-studentt-seed-1111 --work_dir /tanData/mgattn/mgk-2keys-rbf-pi-reg-0-nhead-klen-4head-studentt-seed-1111 --pi_reg 0.0


CUDA_VISIBLE_DEVICES=2,3 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 4 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 200 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --update_mode 'hard2keys' --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name mgk-2keys-hard-pi-reg-0-nhead-klen-4head-diffbwk2small3x-seed-1111 --work_dir /tanData/mgattn/mgk-2keys-hard-pi-reg-0-nhead-klen-4head-diffbwk2small3x-seed-1111 --pi_reg 0.0

CUDA_VISIBLE_DEVICES=4,5 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 4 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 200 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --update_mode 'rbf' --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name mgk-rbf-pi-reg-0-nhead-klen-4head-diffbwk2small3x-seed-1111 --work_dir /tanData/mgattn/mgk-rbf-pi-reg-0-nhead-klen-4head-diffbwk2small3x-seed-1111 --pi_reg 0.0


CUDA_VISIBLE_DEVICES=6,7 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 4 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 200 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --update_mode 'hard' --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name mgk-hard-pi-reg-0-nhead-klen-4head-diffbwk2small3x-seed-1111 --work_dir /tanData/mgattn/mgk-hard-pi-reg-0-nhead-klen-4head-diffbwk2small3x-seed-1111 --pi_reg 0.0


CUDA_VISIBLE_DEVICES=2,3 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 200 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --update_mode 'rbf2keys' --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name mgk-2keys-rbf-pi-reg-0-nhead-klen-8head-diffbwk2small3x-seed-1111 --work_dir /tanData/mgattn/mgk-2keys-rbf-pi-reg-0-nhead-klen-8head-diffbwk2small3x-seed-1111 --pi_reg 0.0



CUDA_VISIBLE_DEVICES=2,3 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 200 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --update_mode 'rbf2keys' --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name mgk-2keys-rbf-pi-reg-0-nhead-klen-8head-seed-1111 --work_dir /tanData/mgattn/mgk-2keys-rbf-pi-reg-0-nhead-klen-8head-seed-1111 --pi_reg 0.0





CUDA_VISIBLE_DEVICES=0,1 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 4 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 200 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --update_mode 'rbf2keys' --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name mgk-2keys-rbf-pi-reg-0-nhead-klen-4head-diffbwk2small3x-pir-0-0001-seed-1111 --work_dir /tanData/mgattn/mgk-2keys-rbf-pi-reg-0-nhead-klen-4head-diffbwk2small3x-pir-0-0001-seed-1111 --pi_reg 0.0001



CUDA_VISIBLE_DEVICES=4,5 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 4 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 200 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --update_mode 'rbf2keys' --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name mgk-2keys-rbf-pi-reg-0-nhead-klen-4head-diffbwk2small3x-pir-0-001-seed-1111 --work_dir /tanData/mgattn/mgk-2keys-rbf-pi-reg-0-nhead-klen-4head-diffbwk2small3x-pir-0-001-seed-1111 --pi_reg 0.001


CUDA_VISIBLE_DEVICES=6,7 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 4 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 200 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --update_mode 'rbf2keys' --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name mgk-2keys-rbf-pi-reg-0-nhead-klen-4head-diffbwk2small3x-pir-0-01-seed-1111 --work_dir /tanData/mgattn/mgk-2keys-rbf-pi-reg-0-nhead-klen-4head-diffbwk2small3x-pir-0-01-seed-1111 --pi_reg 0.01

CUDA_VISIBLE_DEVICES=6,7 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 4 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 200 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --update_mode 'rbf2keys' --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name mgk-2keys-rbf-pi-reg-0-nhead-klen-4head-diffbwk2small3x-clamp-pi-seed-1111 --work_dir /tanData/mgattn/mgk-2keys-rbf-pi-reg-0-nhead-klen-4head-diffbwk2small3x-clamp-pi-seed-1111 --pi_reg 0.0






CUDA_VISIBLE_DEVICES=0,1 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 4 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 200 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --update_mode 'soft2keys' --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name mgk-2keys-soft-pi-reg-0-nhead-klen-4head-diffbwk2small3x-seed-1111 --work_dir /tanData/mgattn/mgk-2keys-soft-pi-reg-0-nhead-klen-4head-diffbwk2small3x-seed-1111 --pi_reg 0.0


CUDA_VISIBLE_DEVICES=4,5 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 200 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --update_mode 'soft2keys' --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name mgk-2keys-soft-pi-reg-0-nhead-klen-8head-diffbwk2small3x-seed-1111 --work_dir /tanData/mgattn/mgk-2keys-soft-pi-reg-0-nhead-klen-8head-diffbwk2small3x-seed-1111 --pi_reg 0.0










CUDA_VISIBLE_DEVICES=2,3 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 4 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 200 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --update_mode 'soft2keys' --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name mgk-2keys-soft-pi-reg-0-nhead-klen-4head-seed-1111 --work_dir /tanData/mgattn/mgk-2keys-soft-pi-reg-0-nhead-klen-4head-seed-1111 --pi_reg 0.0


CUDA_VISIBLE_DEVICES=2,3 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 4 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 200 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --update_mode 'rbf2keys' --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name mgk-2keys-rbf-uniinit-pi-reg-0-nhead-klen-4head-seed-1111 --work_dir /tanData/mgattn/mgk-2keys-rbf-uniinit-pi-reg-0-nhead-klen-4head-seed-1111 --pi_reg 0.0 --md_reg 0.0


CUDA_VISIBLE_DEVICES=2,3 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 4 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 200 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --update_mode 'rbf' --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name mgk-mulgauss-rbf-pi-reg-0-md-reg-0-01-nhead-klen-4head-seed-1111 --work_dir /tanData/mgattn/mgk-mulgauss-rbf-pi-reg-0-md-reg-0-01-nhead-klen-4head-seed-1111 --pi_reg 0.0 --md_reg 0.01



CUDA_VISIBLE_DEVICES=2,3 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 4 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 200 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --update_mode 'rbf' --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name debug --work_dir /tanData/mgattn/debug --pi_reg 0.0 --md_reg 0.01




CUDA_VISIBLE_DEVICES=0,1 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 2 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --multi_gpu --work_dir /tanData/mgattn/debug


CUDA_VISIBLE_DEVICES=0,1 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 200 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --update_mode 'soft' --multi_gpu --work_dir /tanData/mgattn/debug


CUDA_VISIBLE_DEVICES=6,7 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 200 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --update_mode 'rbf' --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name debug1 --work_dir /tanData/mgattn/debug1 --log-interval 1



CUDA_VISIBLE_DEVICES=0 python ./src/eval_sliding_window.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --split 'test' --batch_size 1 --tgt_len 256 --mem_len 0 --clamp_len 256 --work_dir /tanData/mgattn/mgk-2keys-rbf-pi-reg-0-nhead-klen-4head-diffbwk2small3x-clamp-pi-seed-1111

CUDA_VISIBLE_DEVICES=2 python ./src/eval_sliding_window.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --split 'valid' --batch_size 1 --tgt_len 256 --mem_len 0 --clamp_len 256 --work_dir /tanData/mgattn/mgk-2keys-rbf-pi-reg-0-nhead-klen-4head-diffbwk2small3x-clamp-pi-seed-1111

CUDA_VISIBLE_DEVICES=4 python ./src/eval_sliding_window.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --split 'test' --batch_size 1 --tgt_len 256 --mem_len 0 --clamp_len 256 --work_dir /tanData/mgattn/mgk-2keys-rbf-pi-reg-0-nhead-klen-8head-diffbwk2small3x-seed-1111

CUDA_VISIBLE_DEVICES=0 python ./src/eval_sliding_window.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --split 'valid' --batch_size 1 --tgt_len 256 --mem_len 0 --clamp_len 256 --work_dir /tanData/mgattn/mgk-2keys-rbf-pi-reg-0-nhead-klen-8head-diffbwk2small3x-seed-1111



CUDA_VISIBLE_DEVICES=0 python ../src/eval_sliding_window.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --split 'test' --batch_size 1 --tgt_len 256 --mem_len 0 --clamp_len 256 --work_dir {model_dir}



##### MFA Attn ########

CUDA_VISIBLE_DEVICES=2,3 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 201 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name mfaattn-scaled-dot-product --work_dir /tanData/mgattn/mfaattn-scaled-dot-product


CUDA_VISIBLE_DEVICES=0,1 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 202 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --update_mode 'uniform' --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name gmm-gaussian-baseline --work_dir /tanData/mgattn/gmm-gaussian-baseline


CUDA_VISIBLE_DEVICES=4,5 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 202 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --update_mode 'soft' --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name gmm-soft-pi-each-qi --work_dir /tanData/mgattn/gmm-soft-pi-each-qi

CUDA_VISIBLE_DEVICES=2,3 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 202 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --update_mode 'soft' --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name gmm-soft-pi-each-qi-beta-0-99 --work_dir /tanData/mgattn/gmm-soft-pi-each-qi-beta-0-99

CUDA_VISIBLE_DEVICES=4,5 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 202 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --update_mode 'soft' --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name gmm-soft-pi-beta-0-99 --work_dir /tanData/mgattn/gmm-soft-pi-beta-0-99


CUDA_VISIBLE_DEVICES=6,7 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 2 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name softmax-baseline --work_dir /tanData/mgattn/softmax-baseline



CUDA_VISIBLE_DEVICES=2,3 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 202 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --update_mode 'learn' --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name gmm-learn-pi-each-qi --work_dir /tanData/mgattn/gmm-learn-pi-each-qi



CUDA_VISIBLE_DEVICES=2,3 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 202 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --update_mode 'learn' --multi_gpu --project_name 'mgk' --seed 1111 --job_name gmm-learn-pi-each-qi --work_dir /tanData/mgattn/debug


CUDA_VISIBLE_DEVICES=0,1 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 202 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --update_mode 'uniform' --multi_gpu --project_name 'mgk' --seed 1111 --job_name gmm-gaussian-baseline --work_dir /tanData/mgattn/debug


CUDA_VISIBLE_DEVICES=0,1 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 202 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --update_mode 'soft_klen' --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name gmm-soft-klen-scale-beta-0-99 --work_dir /tanData/mgattn/gmm-soft-klen-scale-beta-0-99


CUDA_VISIBLE_DEVICES=2,3 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 202 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --update_mode 'soft_klen_full_learn' --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name gmm-soft-klen-full-learn-beta-0-99 --work_dir /tanData/mgattn/gmm-soft-klen-full-learn-beta-0-99


CUDA_VISIBLE_DEVICES=4,5 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 202 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --update_mode 'soft_klen_scale_learn' --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name gmm-soft-klen-scale-learn-beta-0-99 --work_dir /tanData/mgattn/gmm-soft-klen-scale-learn-beta-0-99


CUDA_VISIBLE_DEVICES=6,7 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 202 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --update_mode 'soft_klen_learn' --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name gmm-soft-klen-learn-beta-0-99 --work_dir /tanData/mgattn/gmm-soft-klen-learn-beta-0-99




CUDA_VISIBLE_DEVICES=0,1 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 202 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --update_mode 'learn_proj' --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name gmm-learn-proj-beta-0-99 --work_dir /tanData/mgattn/gmm-learn-proj-beta-0-99


CUDA_VISIBLE_DEVICES=2,3 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 202 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --update_mode 'learn_proj_batch' --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name gmm-learn-proj-batch-beta-0-99 --work_dir /tanData/mgattn/gmm-learn-proj-batch-beta-0-99


CUDA_VISIBLE_DEVICES=4,5 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 202 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --update_mode 'learn_proj_batch' --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name gmm-learn-proj-batch-beta-0-99-v2 --work_dir /tanData/mgattn/gmm-learn-proj-batch-beta-0-99-v2



CUDA_VISIBLE_DEVICES=6,7 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 202 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --update_mode 'soft_klen_full' --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name gmm-soft-pi-beta-0-99999999 --work_dir /tanData/mgattn/gmm-soft-pi-beta-0-99999999




CUDA_VISIBLE_DEVICES=2 python ./src/eval_sliding_window.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --split 'test' --batch_size 1 --tgt_len 256 --mem_len 0 --clamp_len 256 --work_dir /tanData/mgattn/gmm-soft-pi-beta-0-9999

CUDA_VISIBLE_DEVICES=3 python ./src/eval_sliding_window.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --split 'valid' --batch_size 1 --tgt_len 256 --mem_len 0 --clamp_len 256 --work_dir /tanData/mgattn/gmm-soft-pi-beta-0-9999


CUDA_VISIBLE_DEVICES=2 python ./src/eval_sliding_window.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --split 'test' --batch_size 1 --tgt_len 256 --mem_len 0 --clamp_len 256 --work_dir /tanData/mgattn/gmm-gaussian-baseline

CUDA_VISIBLE_DEVICES=3 python ./src/eval_sliding_window.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --split 'valid' --batch_size 1 --tgt_len 256 --mem_len 0 --clamp_len 256 --work_dir /tanData/mgattn/gmm-gaussian-baseline




# Fourier

CUDA_VISIBLE_DEVICES=0,1 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 2 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name softmax-seed-1111 --work_dir /tanData/mgattn/softmax-seed-1111


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 202 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --update_mode 'learn' --multi_gpu --project_name 'mgk' --seed 1111 --job_name gmm-learn-pi-each-qi --work_dir /tanData/mgattn/debug

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 203 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 16 --multi_gpu --project_name 'mgk' --seed 1111 --job_name fourier --work_dir /tanData/mgattn/debug


CUDA_VISIBLE_DEVICES=2,3 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 204 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --kernel_size 16 8 --stride 16 8 --multi_gpu --project_name 'mgk' --seed 1111 --job_name fourier --work_dir /tanData/mgattn/debug

CUDA_VISIBLE_DEVICES=2,3 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 4 --d_head 32 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 204 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --kernel_size 4 2 --stride 4 2 --multi_gpu --project_name 'mgk' --seed 1111 --job_name fourier --work_dir /tanData/mgattn/debug


# Hierarchical Dirichlet Process Transformer
CUDA_VISIBLE_DEVICES=6,7 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 2 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --multi_gpu --project_name 'mgk' --seed 1111 --job_name softmax-seed-1111 --work_dir /tanData/mgattn/debug


CUDA_VISIBLE_DEVICES=6,7 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 205 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --n_global_head 4 --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name hdp-n-global-head-4 --work_dir /tanData/mgattn/hdp-n-global-head-4


CUDA_VISIBLE_DEVICES=0,1 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 205 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --n_global_head 5 --multi_gpu --project_name 'mgk' --seed 1111 --job_name hdp-n-global-head-5-q-k --work_dir /tanData/mgattn/debug


CUDA_VISIBLE_DEVICES=6,7 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 205 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --n_global_head 5 --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name hdp-n-global-head-5-qk-inside-softmax --work_dir /tanData/mgattn/hdp-n-global-head-5-qk-inside-softmax


CUDA_VISIBLE_DEVICES=4,5 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 205 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --n_global_head 5 --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name hdp-n-global-head-5-k --work_dir /tanData/mgattn/hdp-n-global-head-5-k


CUDA_VISIBLE_DEVICES=0,1 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 205 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --n_global_head 5 --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name hdp-n-global-head-5-q --work_dir /tanData/mgattn/hdp-n-global-head-5-q


CUDA_VISIBLE_DEVICES=6,7 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 205 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --n_global_head 2 --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name hdp-n-global-head-2-qk-inside-softmax --work_dir /tanData/mgattn/hdp-n-global-head-2-qk-inside-softmax


CUDA_VISIBLE_DEVICES=2,3 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 12 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 205 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --n_global_head 5 --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name hdp-n-global-head-5-local-head-12-q --work_dir /tanData/mgattn/hdp-n-global-head-5-local-head-12-q




CUDA_VISIBLE_DEVICES=0,1 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 12 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 205 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --n_global_head 2 --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name hdp-n-global-head-2-local-head-12-q --work_dir /tanData/mgattn/hdp-n-global-head-2-local-head-12-q



CUDA_VISIBLE_DEVICES=0,1 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 205 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --n_global_head 4 --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name hdp-n-global-head-4-qk --work_dir /tanData/mgattn/hdp-n-global-head-4-qk


CUDA_VISIBLE_DEVICES=2,3 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 205 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --n_global_head 4 --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name hdp-n-global-head-4-qk-nonlin --work_dir /tanData/mgattn/hdp-n-global-head-4-qk-nonlin



CUDA_VISIBLE_DEVICES=4,5 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 205 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --n_global_head 4 --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name hdp-n-global-head-4-qk-layernorm --work_dir /tanData/mgattn/hdp-n-global-head-4-qk-layernorm



CUDA_VISIBLE_DEVICES=4,5 python ./src/train.py --cuda --data /tanData/nlp_data/wikitext-103/ --dataset wt103 --adaptive --n_layer 16 --d_model 128 --n_head 8 --d_head 16 --d_inner 2048 --dropout 0.1 --dropatt 0.0 --optim adam --lr 0.00025 --warmup_step 2000 --max_step 500000 --attn_type 205 --tgt_len 256 --mem_len 0 --eval_tgt_len 256 --batch_size 96 --n_global_head 4 --multi_gpu --use_wandb --project_name 'mgk' --seed 1111 --job_name hdp-n-global-head-4-qk-nonlin-gelu --work_dir /tanData/mgattn/hdp-n-global-head-4-qk-nonlin-gelu