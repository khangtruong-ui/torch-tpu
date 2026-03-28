import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from datasets import load_dataset
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

# ==== CONFIG ====
SAVE_PATH = "/app/saved_weights/"
os.makedirs(SAVE_PATH, exist_ok=True)
BATCH_SIZE = 1024
EPOCHES = 20
LR = 1e-4
NUM_WORKERS = 32  # As requested

# ==== UTILS ====
def get_label_from_filename(filename):
    # Extracts "circular_farmland" from "circular_farmland_055.jpg"
    return "_".join(filename.split('_')[:-1])

class NWPUDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, label2idx, transform=None):
        self.dataset = hf_dataset
        self.label2idx = label2idx
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index % len(self.dataset)]
        image = item["image"].convert("RGB")
        label_str = get_label_from_filename(item["raw_filename"])
        label = self.label2idx[label_str]

        if self.transform:
            image = self.transform(image)

        return image, label

# ==== TRAINING FUNCTION ====
def train_fn(index):
    device = xm.xla_device()

    # 1. Load HF Dataset & Map Labels
    # This happens inside train_fn so each process has access
    raw_dataset = load_dataset("KhangTruong/NWPU_Split")
    unique_labels = sorted(list(set(get_label_from_filename(f) for f in raw_dataset['train']['raw_filename'])))
    label2idx = {label: i for i, label in enumerate(unique_labels)}
    num_classes = len(unique_labels)

    # 2. Preprocessing & DataLoaders
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = NWPUDataset(raw_dataset['train'], label2idx, transform=preprocess)
    
    sampler = torch.utils.data.distributed.DistributedSampler(
        train_ds,
        num_replicas=torch_xla.runtime.world_size(),
        rank=torch_xla.runtime.global_ordinal(),
        shuffle=True
    )

    loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
        drop_last=True,
        prefetch_factor=1,
    )

    # Wrap the loader for XLA (Crucial to prevent stalling)
    mp_loader = pl.MpDeviceLoader(loader, device)

    # 3. Model Setup
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    
    # Sync parameters across cores
    xm.broadcast_master_param(model)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # 4. Training Loop
    model.train()
    xm.master_print(f"Starting training with {num_classes} classes...")

    for epoch in range(EPOCHES):
        for batch_idx, (data, target) in enumerate(tqdm(mp_loader) if xm.is_master_ordinal() else mp_loader):
            optimizer.zero_grad()
            
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient reduction and weight update
            xm.optimizer_step(optimizer)

        xm.master_print(f"Epoch {epoch + 1} Batch {batch_idx} | Loss {loss.item():.4f}")

    # 5. Save Weights
    # Only save from the master process to avoid file corruption
    xm.save(model.state_dict(), os.path.join(SAVE_PATH, "resnet_nwpu.pth"))
    xm.master_print(f"Training finished. Weights saved to {SAVE_PATH}")

if __name__ == "__main__":
    # Launch for all 8 TPU cores
    xmp.spawn(train_fn, args=(), start_method='fork')