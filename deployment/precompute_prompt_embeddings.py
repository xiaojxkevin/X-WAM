"""Offline T5 prompt-embedding pre-encoding for deployment on small GPUs.

Encodes every task in a LeRobot ``meta/tasks.jsonl`` with the Wan2.2 umt5-xxl T5
encoder (the exact ``modules/t5.py`` stack used at training time) and saves:

    <dst>/
        prompt_embeddings.pt   # {"prompts": [str], "embeddings": bf16 [N, 512, 4096]}

At serving time the policy loads this file and never touches T5 again (saves
~11 GB weights + per-request encode latency). The saved embeddings are
bit-identical to what ``T5EncoderModel(prompts)`` would produce.

Run (CPU is fine, ~1 min; GPU faster)::

    .venv/bin/python deployment/precompute_prompt_embeddings.py \
        --tasks-jsonl raw_data/multitask_merged_v1/meta/tasks.jsonl \
        --wan-checkpoint-dir ./checkpoints/Wan2.2-TI2V-5B \
        --dst experiments/multitask_merged-v1-sft
"""

import argparse
import json
import logging
import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from omegaconf import OmegaConf

from modules.t5 import T5EncoderModel


def load_tasks(tasks_jsonl: str) -> list[str]:
    prompts = []
    with open(tasks_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            prompts.append(rec["task"])
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks-jsonl", type=str, required=True, help="LeRobot meta/tasks.jsonl")
    parser.add_argument("--wan-checkpoint-dir", type=str, default="./checkpoints/Wan2.2-TI2V-5B")
    parser.add_argument("--dst", type=str, default=None, help="Output dir (default: alongside exp config).")
    parser.add_argument("--config", type=str, default=None, help="config.yaml for text_len/t5 settings.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.config:
        config = OmegaConf.load(args.config)
        text_len = int(config.text_len)
        t5_checkpoint = config.get("t5_checkpoint", "models_t5_umt5-xxl-enc-bf16.pth")
        t5_tokenizer = config.get("t5_tokenizer", "google/umt5-xxl")
        t5_dtype = eval(config.get("t5_dtype", "torch.bfloat16"))
    else:
        text_len, t5_checkpoint, t5_tokenizer, t5_dtype = 512, "models_t5_umt5-xxl-enc-bf16.pth", "google/umt5-xxl", torch.bfloat16

    prompts = load_tasks(args.tasks_jsonl)
    if not prompts:
        raise ValueError(f"no tasks found in {args.tasks_jsonl}")
    logging.info("Tasks (%d): %s", len(prompts), prompts)

    encoder = T5EncoderModel(
        text_len=text_len,
        dtype=t5_dtype,
        device=torch.device(args.device),
        checkpoint_path=os.path.join(args.wan_checkpoint_dir, t5_checkpoint),
        tokenizer_path=os.path.join(args.wan_checkpoint_dir, t5_tokenizer),
    )
    encoder.eval()

    with torch.inference_mode():
        embeddings = encoder(prompts)  # [N, text_len, 4096], t5_dtype

    embeddings = embeddings.to(torch.bfloat16).cpu()
    logging.info("Encoded %d prompts -> %s", len(prompts), tuple(embeddings.shape))

    dst = args.dst or os.path.dirname(os.path.abspath(args.tasks_jsonl))
    os.makedirs(dst, exist_ok=True)
    out_path = os.path.join(dst, "prompt_embeddings.pt")
    torch.save({"prompts": prompts, "embeddings": embeddings}, out_path)
    logging.info("Saved %s (%.1f MB)", out_path, os.path.getsize(out_path) / 1e6)


if __name__ == "__main__":
    main()
