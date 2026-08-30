#!/bin/bash

# place_fruits_in_bucket
torchrun --nnodes=1 --node_rank=0 --nproc_per_node=8 \
    --master_addr=localhost --master_port=29500 \
    scripts/train_sft.py \
    dataset=place_fruits \
    exp_name="${EXP_NAME}" \
    wan_checkpoint_dir=./checkpoints/Wan2.2-TI2V-5B \
    use_wandb=true \
    wandb_project="d4rt_vla_baselines" \
    wandb_run_id="place_fruits_in_bucket-v1"
