import logging
import json
import io
import os
import time

import boto3
import tempfile

import torch
import torchaudio
from tortoise import api
from tortoise.utils import audio
from tortoise.utils.text import split_and_recombine_text
from tortoise.models.autoregressive import UnifiedVoice

#create logger for sagemaker
logger = logging.getLogger(__name__)
HALF=True
KV_CACHE=True
USE_DEEPSPEED=False

MODEL_DIR = os.getenv('SM_MODEL_DIR', '/opt/ml/model')

def model_fn(model_dir):
    """
    Load the model for inference
    """
    # List all the folders located in '/ml/opt/model
    model_folders = os.listdir(model_dir)
    logger.info(f"Found the following models: {model_folders}")
    
    
    logger.info("Loading model")
    model = api.TextToSpeech(half=HALF, kv_cache=KV_CACHE, use_deepspeed=USE_DEEPSPEED, models_dir=model_dir)
    logger.info("Model loaded")
    
    return model

def download_get_conditioning_latents(model, voice_samples):
    """
    Download voice samples from S3 and compute conditioning latents

    Args:
        model (Tortoise): The Tortoise model to load the weights into.
        voice_samples_s3_uri (str): The S3 URI for the voice samples.
    """

    logger.info("load voice samples")

    # Load all voice sample files
    voice_samples_dir = f"samples/{voice_samples}"
    voice_samples = []
    for filename in os.listdir(voice_samples_dir):
        voice_samples.append(audio.load_audio(os.path.join(voice_samples_dir, 
                                                           filename), 
                                              22050))

    logger.info("Computing conditioning latents")
    # Compute conditioning latentstents for the given voice samples.
    conditioning_latents = model.get_conditioning_latents(voice_samples)
    logger.info("Conditioning latents computed")
    
    return conditioning_latents
    
def predict_fn(input_data, model):
    """
    Run prediction on input data
    """
    
    if input_data['voice_samples']:
        conditioning_latents = download_get_conditioning_latents(model, input_data['voice_samples'])
    
    logger.info("Generating with params: %s", input_data)

    # Synthesize
    if input_data['voice_samples']:
        audio_clip = model.tts(input_data['text'], voice_samples=None,
                               conditioning_latents=conditioning_latents).squeeze(0).cpu()
    else:
        audio_clip = model.tts_with_preset(input_data['text'],
                                           preset="fast").squeeze(0).cpu()

    # create a BytesIO object
    buffer = io.BytesIO()

    # Save to temporary file
    torchaudio.save(buffer, audio_clip, 24000, format="wav")

    # Return the butes from the BytesIO object
    return buffer.getvalue()


def input_fn(request_body, request_content_type):
    """
    Deserializes the input request body and prepares data for text-to-speech generation.

    Args:
        request_body (str): A JSON string containing the following fields:

            text (str): The text to be synthesized into speech.
            voice_samples (str): folder directory of voice samples.
            inference_params (dict): A dictionary containing parameters controlling the tortoise-tts generation process.

    Returns:
        A dict containing:
            text (str): The extracted text to be generated.
            voice_samples (str): folder directory of voice samples.
            inference_params (dict): The parsed inference parameters (default: empty dict).

    Raises:
        ValueError: If any required fields are missing or invalid in the request body.
    """

    logger.info('Received input: %s', request_body)
    if request_content_type == "application/json":
        try:
            request = json.loads(request_body)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format in request body")
    else:
        raise ValueError("Unsupported content type: {}".format(request_content_type))

    # Extract and validate required fields
    required_fields = ["text", "voice_samples"]
    missing_fields = [field for field in required_fields if field not in request]
    if missing_fields:
        raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

    # Extract and handle optional fields with defaults
    return {
        "text": request["text"],
        "voice_samples": request.get("voice_samples", None),
        "inference_params": request.get("inference_params", {}),
    }


def output_fn(response_body, response_content_type):

    """
    Serialize and prepare the prediction output
    """

    logger.info('Returning response')
    return {
        "statusCode": 200,
        "body": response_body}