import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

import os
import math
import json
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
from dataset.ImageCaptionDataset import ImageCaptionDataset


from transformers import AutoProcessor, BlipForConditionalGeneration

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


train_dataset = ImageCaptionDataset('./cocoDataset/annotation/annotations/captions_train2014.json', processor)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=2)

device = "cuda" if torch.cuda.is_available() else "CPU"
model.to(device)
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)


min_total_loss = math.inf
for epoch in range(50):
    print("Epoch : ", epoch)
    total_loss = 0
    with tqdm(total=len(train_dataloader)) as pbar:
        for idx, batch in enumerate(train_dataloader):
            input_ids = batch.pop("input_ids").to(device)
            pixel_values = batch.pop("pixel_values").to(device)
            
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                outputs = model(input_ids=input_ids,
                                pixel_values = pixel_values,
                                labels=input_ids)
            
            loss = outputs.loss
            
            pbar.set_postfix(loss = loss.item())
            pbar.update(1)
            total_loss += loss.item()    
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
    if min_total_loss > total_loss:
        torch.save(model.state_dict(), f"./checkpoints/BLIP_trainin_ver-{epoch}-{total_loss:.2f}.pt")
        break