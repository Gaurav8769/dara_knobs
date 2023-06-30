import os
import faiss
import torch
import skimage
import requests
from google.cloud import storage
import requests
import base64
from PIL import Image
import hashlib
import pinecone
import numpy as np
import pandas as pd
from io import BytesIO
import IPython.display
import matplotlib.pyplot as plt
from collections import OrderedDict
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer

def change_url(url):
    new_url = url.replace("photos100", "photos1000")
    # new_url = url.replace("photos1000", "photos1000")
    return new_url
def convert_gs_url_to_https(gs_url):
    gs_prefix = "gs://"
    gs_storage_url = "https://storage.googleapis.com"

    if gs_url.startswith(gs_prefix):
        bucket_and_path = gs_url[len(gs_prefix):]
        return f"{gs_storage_url}/{bucket_and_path}"

    return gs_url
def get_image_from_url(url):
    try:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        return image
    except Exception as e:
        print("Error retrieving image:", e)
        return None
def get_model_info(model_ID, device):
  # Save the model to device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
	model = CLIPModel.from_pretrained(model_ID).to(device)
 	# Get the processor
	processor = CLIPProcessor.from_pretrained(model_ID)
  # Get the tokenizer
	tokenizer = CLIPTokenizer.from_pretrained(model_ID)
       # Return model, processor & tokenizer
	return model, processor, tokenizer
def get_single_image_embedding(my_image):
  image = processor(
		text = None,
		images = my_image,
		return_tensors="pt"
		)["pixel_values"].to(device)
  embedding = model.get_image_features(image)
  embedding_as_np = embedding.cpu().detach().numpy()
  return embedding_as_np
def get_single_text_embedding(text):
  inputs = tokenizer(text, return_tensors = "pt").to(device)
  text_embeddings = model.get_text_features(**inputs)
  # convert the embeddings to numpy array
  embedding_as_np = text_embeddings.cpu().detach().numpy()
  return embedding_as_np
def get_all_text_embeddings(df, text_col):
  df["text_embeddings"] = df[str(text_col)].apply(get_single_text_embedding)
  return df
