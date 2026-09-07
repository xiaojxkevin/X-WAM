#!/bin/bash

# multitask_merged (5 tasks: pens / towel / mugs / fruits / bowls)
torchrun --nnodes=1 --node_rank=0 --nproc_per_node=8 \
    --master_addr=localhost --master_port=29500 \
    scripts/train_sft.py \
    dataset=multitask_merged \
    exp_name="${EXP_NAME}" \
    wan_checkpoint_dir=./checkpoints/Wan2.2-TI2V-5B \
    use_wandb=true \
    wandb_project="xwam" \
    wandb_run_id="multitask_merged-v1-sft"
