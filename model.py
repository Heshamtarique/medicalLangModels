!pip install -q -U transformers==4.37.2
!pip install -q bitsandbytes==0.41.3 accelerate==0.25.0
!pip install -q git+https://github.com/openai/whisper.git
!pip install -q gradio
!pip install -q gTTS

import torch
from transformers import BitsAndBytesConfig, pipeline

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model_id = "llava-hf/llava-1.5-7b-hf"

pipe = pipeline("image-to-text",
                model=model_id,
                model_kwargs={"quantization_config": quantization_config})
import whisper
import gradio as gr
import time
import warnings
import os
from gtts import gTTS

from PIL import Image
import os

if not os.path.exists(image_path):
    raise ValueError("File path does not exist:", image_path)



from PIL import Image, UnidentifiedImageError

import requests
from PIL import Image
from io import BytesIO

def load_image_from_url(url):
    try:
        # Send a HTTP request to the specified URL
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful

        # Use BytesIO to create an in-memory byte stream from the response content
        image_data = BytesIO(response.content)

        # Open the image using PIL
        image = Image.open(image_data)
        return image
    except requests.RequestException as e:
        print(f"An error occurred while fetching the image: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return None

# Example usage
image_url = "https://img.lb.wbmdstatic.com/vim/live/webmd/consumer_assets/site_images/articles/health_tools/guide_to_unusual_skin_conditions_slideshow/1800ss_medical_images_rm_cutaneous_amyloidosis-1.jpg?resize=652px:*&output-quality=100"
image = load_image_from_url(image_url)

# If the image is successfully loaded, display it
if image:
    image.show()


image

















