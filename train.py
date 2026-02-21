
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import math
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer
from model import ModernGPT
from config import GPTConfig

# =============================
# CONFIG
# =============================
TOTAL_TOKENS_TARGET = 200_000_000
SEQ_LEN = 512
BATCH_SIZE = 32
CHECKPOINT_DIR = "nodegpt50/checkpoints"
TOKENIZER_PATH = "nodegpt50/tokenizer.json"

# =============================
# DDP SETUP
# =============================
def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

# =============================
# STREAMING DATASET
# =============================
class StreamingTinyStories(Dataset):
    def __init__(self, tokenizer, seq_len):
        self.dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.buffer = []
        self.iterator = iter(self.dataset)

    def __iter__(self):
        return self

    def __next__(self):
        while len(self.buffer) < self.seq_len + 1:
            sample = next(self.iterator)
            ids = self.tokenizer.encode(sample["text"]).ids
            self.buffer.extend(ids)

        chunk = self.buffer[:self.seq_len + 1]
        self.buffer = self.buffer[self.seq_len:]

        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

# =============================
# CHECKPOINTING
# =============================
def save_checkpoint(model, optimizer, tokens_seen, epoch):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, "latest.pt")
    torch.save({
        "model": model.module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "tokens_seen": tokens_seen,
        "epoch": epoch
    }, path)

def load_checkpoint(model, optimizer):
    path = os.path.join(CHECKPOINT_DIR, "latest.pt")
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location="cpu")
        model.module.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        return checkpoint["tokens_seen"], checkpoint["epoch"]
    return 0, 0

# =============================
# TRAIN LOOP (DDP + AMP + MICRO BATCH)
# =============================
def main():
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

    config = GPTConfig()
    model = ModernGPT(config).to(device)
    model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    scaler = torch.cuda.amp.GradScaler()

    tokens_seen = 0
    epoch = 0

    tokens_seen, epoch = load_checkpoint(model, optimizer)

    dataset = StreamingTinyStories(tokenizer, SEQ_LEN)

    print(f"[Rank {local_rank}] Starting training from {tokens_seen} tokens")

    MICRO_BATCH_SIZE = 16  # increase this to 24 or 32 if GPU memory allows

    while tokens_seen < TOTAL_TOKENS_TARGET:
        epoch += 1
        step = 0
        start_time = time.time()

        batch_x = []
        batch_y = []

        for x, y in dataset:
            batch_x.append(x)
            batch_y.append(y)

            if len(batch_x) < MICRO_BATCH_SIZE:
                continue

            x_batch = torch.stack(batch_x).to(device)
            y_batch = torch.stack(batch_y).to(device)

            optimizer.zero_grad(set_to_none=True)

            # ---- Mixed Precision Forward ----
            with torch.cuda.amp.autocast():
                logits = model(x_batch)
                loss = criterion(
                    logits.view(-1, logits.size(-1)),
                    y_batch.view(-1)
                )

            # ---- Mixed Precision Backward ----
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            tokens_seen += SEQ_LEN * MICRO_BATCH_SIZE
            step += 1

            batch_x = []
            batch_y = []

            if local_rank == 0 and step % 100 == 0:
                elapsed = time.time() - start_time
                print(
                    f"Epoch {epoch} | Step {step} | "
                    f"Loss {loss.item():.4f} | "
                    f"Tokens {tokens_seen} | "
                    f"{tokens_seen/elapsed:.0f} tok/s"
                )

            if tokens_seen >= TOTAL_TOKENS_TARGET:
                break

        if local_rank == 0:
            save_checkpoint(model, optimizer, tokens_seen, epoch)

    if local_rank == 0:
        print("Training complete.")

if __name__ == "__main__":
    main()
