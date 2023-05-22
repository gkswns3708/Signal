import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoProcessor
from transformers import BlipForConditionalGeneration

from datasets import load_dataset

from dataset.ImageCaptioningDataset import ImageCaptioningDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

# preparing datasets
dataset = load_dataset("ybelkada/football-dataset", split="train")

# TODO : AutoProcessor의 기능
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# TODO : BlipForConditionalGeneration의 내부 아키텍쳐(간략히)
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
model = model.to(device)

# Dataset & DataLoader
inference_dataset = ImageCaptioningDataset(dataset, processor)
inference_dataloader = DataLoader(inference_dataset, shuffle=True, batch_size=1)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)


for idx, example in enumerate(dataset):    
    image = example["image"]
    inputs = processor(images=image, return_tensors="pt").to(device)
    pixel_values = inputs.pixel_values
    
    # TODO : model.generate의 API 분석
    generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print(generated_caption)