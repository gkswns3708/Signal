from fastapi import FastAPI, UploadFile, File

import torch
from PIL import Image

from transformers import AutoProcessor
from transformers import BlipForConditionalGeneration

from google.cloud import texttospeech

app = FastAPI()

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# TODO : BlipForConditionalGeneration의 내부 아키텍쳐(간략히)
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
device = "cuda" if torch.cuda.is_available() else "cpu"

@app.post("/image_process")
async def process_image(file: UploadFile = File(...)):
    image = Image.open(file.file)
    
    image = image.resize((224, 224))
    inputs = processor(images=image, return_tensors="pt").to(device)
    pixel_values = inputs.pixel_values
    
    generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
    generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    client = texttospeech.TextToSpeechClient()
    
    audio_caption = []
    if "shirt" in generated_caption:
        audio_caption.append("셔츠") 
    
    if "black" in generated_caption:
        audio_caption.append("검은색") 
    
    synthesis_input = texttospeech.SynthesisInput(text="횡단보도, 신호등")
    voice = texttospeech.VoiceSelectionParams(language_code="ko-KR", ssml_gender=texttospeech.SsmlVoiceGender.FEMALE)    
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
    with open("generated_caption_audio.mp3", "wb") as out:
    # Write the response to the output file.
        out.write(response.audio_content)
        print('Audio content written to file "generated_caption_audio.mp3"')
    
    return {"message" : 'Audio content written to file "generated_caption_audio.mp3"'}