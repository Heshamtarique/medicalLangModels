from main import *
import nltk
nltk.download('punkt')
from nltk import sent_tokenize



import locale
print(locale.getlocale())  # Before running the pipeline
# Run the pipeline
print(locale.getlocale())  # After running the pipeline

max_new_tokens = 200


prompt_instructions = """
Describe the image using as much detail as possible,
is it a painting, a photograph, what colors are predominant,
what is the image about?
"""

prompt = "USER: <image>\n" + prompt_instructions + "\nASSISTANT:"

outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})
# outputs
# print(outputs[0]["generated_text"])
for sent in sent_tokenize(outputs[0]["generated_text"]):
    print(sent)

warnings.filterwarnings("ignore")

import warnings
from gtts import gTTS
import numpy as np

torch.cuda.is_available()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using torch {torch.__version__} ({DEVICE})")

import whisper
model = whisper.load_model("medium", device=DEVICE)
print(
    f"Model is {'multilingual' if model.is_multilingual else 'English-only'} "
    f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
)


import re

input_text = 'What could be the issue on the skin?'
input_image = image

# load the image
# image = Image.open(input_image)

# prompt_instructions = """
# Describe the image using as much detail as possible, is it a painting, a photograph, what colors are predominant, what is the image about?
# """

# print(input_text)
prompt_instructions = """
Act as an expert in imagery descriptive analysis, using as much detail as possible from the image, respond to the following prompt:
""" + input_text
prompt = "USER: <image>\n" + prompt_instructions + "\nASSISTANT:"

# print(prompt)

outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})

match = re.search(r'ASSISTANT:\s*(.*)', outputs[0]["generated_text"])

if match:
    # Extract the text after "ASSISTANT:"
    extracted_text = match.group(1)
    print(extracted_text)
else:
    print("No match found.")

for sent in sent_tokenize(outputs[0]["generated_text"]):
    print(sent)




'''
In the image, there is a person with a red, pink, or reddish-brown mark on their face, which appears to be a skin condition or an injury. The mark is located on the person's cheek, and it is described as a "pimple" or a "red spot." The presence of this mark suggests that the person might be experiencing skin irritation, inflammation, or an infection. It is important to consult a dermatologist or a healthcare professional to determine the cause of the mark and receive appropriate treatment.
USER:  

Act as an expert in imagery descriptive analysis, using as much detail as possible from the image, respond to the following prompt:
What could be the issue on the skin?
ASSISTANT: In the image, there is a person with a red, pink, or reddish-brown mark on their face, which appears to be a skin condition or an injury.
The mark is located on the person's cheek, and it is described as a "pimple" or a "red spot."
The presence of this mark suggests that the person might be experiencing skin irritation, inflammation, or an infection.
It is important to consult a dermatologist or a healthcare professional to determine the cause of the mark and receive appropriate treatment.
'''















