#!/usr/bin/env python
# coding=utf-8
import argparse
import logging
import math
import os
import random
from pathlib import Path
from tqdm import tqdm
from time import time

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version

torch.xla = torch_xla
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="SD Training script for torch_xla.")
    # Arguments maintained exactly as the original snippet
    parser.add_argument("--input_perturbation", type=float, default=0)
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument("--train_data_dir", type=str, default=None)
    parser.add_argument("--image_column", type=str, default="image")
    parser.add_argument("--caption_column", type=str, default="raw")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--validation_prompts", type=str, default=None, nargs="+")
    parser.add_argument("--output_dir", type=str, default="sd-model-finetuned")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--center_crop", default=False, action="store_true")
    parser.add_argument("--random_flip", action="store_true")
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--snr_gamma", type=float, default=None)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--dataloader_num_workers", type=int, default=32)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--prediction_type", type=str, default=None)
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "bf16"])
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--noise_offset", type=float, default=0)
    parser.add_argument("--image_interpolation_mode", type=str, default="lanczos")
    # Redundant but kept for CLI stability
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    
    return parser.parse_args()

def _mp_fn(index, args):
    device = xm.xla_device()
    
    if args.seed is not None:
        xm.set_rng_state(args.seed, device)
        random.seed(args.seed)

    # Load Models
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet").to(device)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.train()

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Dataset Setup
    if args.dataset_name is not None:
        dataset = load_dataset(args.dataset_name, args.dataset_config_name, cache_dir=args.cache_dir)
    else:
        dataset = load_dataset("imagefolder", data_files={"train": os.path.join(args.train_data_dir, "**")})

    # Core transformations
    train_transforms = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
        transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    def collate_fn(examples):
        pixel_values = torch.stack([train_transforms(example[args.image_column].convert("RGB")) for example in examples])
        
        captions = []
        for example in examples:
            caption = example[args.caption_column]
            captions.append(random.choice(caption) if isinstance(caption, list) else caption)
        
        input_ids = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids

        return {"pixel_values": pixel_values, "input_ids": input_ids}

    train_dataset = dataset["train"]
    if args.max_train_samples is not None:
        train_dataset = train_dataset.shuffle(seed=args.seed).select(range(args.max_train_samples))

    # Distributed Sampler for XLA
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(), shuffle=True
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
        persistent_workers=True,
        drop_last=True,
        prefetch_factor=1,
    )

    # Optimization
    optimizer = torch.optim.AdamW(
        unet.parameters(), 
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader))
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Parallel Loader
    train_device_loader = pl.MpDeviceLoader(train_dataloader, device)

    xm.master_print('Hello')
    start_time = time()
    next(iter(train_dataloader))
    xm.master_print(f'Datatime: {time() - start_time}')

    # Training loop
    global_step = 0
    for epoch in range(args.num_train_epochs):
        xm.master_print('Going')
        loader = iter(train_dataloader)
        
        xm.master_print('Start up ...')
        for step in range(len(train_dataloader)):
            print(f'Step {step}')
            batch = next(loader)
            print('Going')
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()

            # VAE encoding
            latents = vae.encode(batch["pixel_values"]).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            # Sample noise
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
            
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            encoder_hidden_states = text_encoder(batch["input_ids"])[0]

            # Predict
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            else:
                target = noise_scheduler.get_velocity(latents, noise, timesteps)

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            loss.backward()
            
            lr_scheduler.step()
            xm.optimizer_step(optimizer)
            global_step += 1

            if global_step >= args.max_train_steps:
                break
    
    # Final save
    xm.rendezvous('final_save')
    if xm.is_master_ordinal():
        unet.to("cpu").save_pretrained(args.output_dir)

def main():
    args = parse_args()
    # nprocs=8 for a standard TPU v3-8/v2-8
    xmp.spawn(_mp_fn, args=(args,), start_method='fork')

if __name__ == "__main__":
    main()