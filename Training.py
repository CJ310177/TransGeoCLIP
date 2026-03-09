import torch
import os
import numpy as np
import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.utils import MP16Dataset
from utils.TransGeoCLIP import TransGeoCLIP
from accelerate import Accelerator, DistributedDataParallelKwargs
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def train_1epoch(dataloader, model, vision_processor, text_processor, optimizer, scheduler, accelerator):
    """Train for one epoch."""
    model.train()
    # Progress bar (main process only)
    t = tqdm(dataloader, disable=not accelerator.is_local_main_process)
    
    for i, (images, texts, longitude, latitude) in enumerate(t):
        # Tokenize text
        texts = text_processor(
            text=texts, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt', 
            max_length=77
        )
        # Move text inputs to device
        texts = {k: v.to(accelerator.device) for k, v in texts.items()}

        # Move other inputs to device
        images = images.to(accelerator.device)
        longitude = longitude.to(accelerator.device).float()
        latitude = latitude.to(accelerator.device).float()
        
        # Zero gradients
        optimizer.zero_grad()
        # Forward pass
        output = model(images, texts, longitude, latitude, return_loss=True)
        loss = output['loss']
   
        # Backward pass
        accelerator.backward(loss)
        optimizer.step()
        
        # Log progress
        if i % 1 == 0 and accelerator.is_local_main_process:
            t.set_description(f'step {i}, loss {loss.item():.4f}, lr {scheduler.get_last_lr()[0]:.6f}')



def main():
    # Create checkpoint directory
    os.makedirs('checkpoints', exist_ok=True)

    # Configure DDP and Accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        kwargs_handlers=[ddp_kwargs],
        split_batches=True  
    )
    
    # Print device info (main process)
    if accelerator.is_local_main_process:
        print(f"Device: {accelerator.device}")
        print(f"Num Processes: {accelerator.num_processes}")
    
    # Initialize model
    model = TransGeoCLIP(accelerator.device).to(accelerator.device)
    
    # Load pre-trained location encoder weights (main process)
    if accelerator.is_local_main_process:
        location_encoder_dict = torch.load(
            'your path',
            map_location=accelerator.device
        )
        model.location_encoder.load_state_dict(location_encoder_dict, strict=False)

    # Get processors and dataset
    vision_processor = model.vision_processor
    text_processor = model.text_processor
    dataset = MP16Dataset(
        vision_processor=vision_processor, 
        text_processor=text_processor
    )
    # Create dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=256, 
        shuffle=False, 
        num_workers=16, 
        pin_memory=True, 
        prefetch_factor=5
    )
    
    # Define optimizer (trainable params only)
    optimizer = torch.optim.AdamW(
        [param for name, param in model.named_parameters() if param.requires_grad],
        lr=3e-5,
        weight_decay=1e-6
    )
    # Define learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=1, 
        gamma=0.87
    )
    
    # Checkpoint resumption logic
    resume_epoch = 0
    
    if accelerator.is_local_main_process:
        # Find latest checkpoint
        for epoch in range(9, -1, -1):
            cp_path = f'your path'
            if os.path.exists(cp_path):
                checkpoint_path = cp_path
                resume_epoch = epoch + 1
                break
        
        # Load checkpoint if found
        if checkpoint_path is not None:
            print(f"Resuming from {checkpoint_path}, starting epoch {resume_epoch}")
            checkpoint = torch.load(checkpoint_path, map_location=accelerator.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Prepare components for distributed training
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )
    
    # Training loop
    for epoch in range(resume_epoch, 10):
        if accelerator.is_local_main_process:
            print(f"\n===== Starting Epoch {epoch+1} =====")
        
        # Train one epoch
        train_1epoch(
            dataloader, 
            model, 
            vision_processor, 
            text_processor, 
            optimizer, 
            scheduler, 
            accelerator
        )
        
        # Save checkpoint (main process)
        if accelerator.is_local_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': unwrapped_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            torch.save(checkpoint, f'checkpoints_TransGeoCLIP_df/TransGeoCLIP_{epoch}_.pth')
            print(f"Epoch {epoch+1} completed, checkpoint saved")

if __name__ == '__main__':
    main()
   
# Note: Full documentation will be updated upon official acceptance.