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
    print(f"Found the following models: {model_folders}")
    
    
    print("Loading model ===============")
    model = api.TextToSpeech(half=HALF, kv_cache=KV_CACHE, use_deepspeed=USE_DEEPSPEED, models_dir=model_dir)
    
    print("Model loaded =================")
    
    return model
    
def predict_fn(input_data, model):
    """
    Run prediction on input data
    """

    samples_dir = os.path.join(MODEL_DIR, 'code/samples')
    
    voice_samples = []

    voice_samples_dir = os.path.join(samples_dir, input_data['voice_id'])

    print(f"List voice sample id dir: {os.listdir(voice_samples_dir)}")

    for filename in os.listdir(voice_samples_dir):
        print(f"processing {filename}")
        voice_samples.append(audio.load_audio(os.path.join(voice_samples_dir, filename), 22050))
    
    conditioning_latents = model.get_conditioning_latents(voice_samples)
    
    print("Generating with params: %s", input_data)

    # Synthesize
    audio_clip = model.tts(input_data['text'], voice_samples=None,
                           conditioning_latents=conditioning_latents).squeeze(0).cpu()

    # create a BytesIO object
    buffer = io.BytesIO()

    # Save to temporary file
    torchaudio.save(buffer, audio_clip, 24000, format="wav")

    # Configure boto3 client
    session = boto3.Session()
    s3_client = session.client('s3')

    # Important: Seek to the beginning of the buffer
    buffer.seek(0)
    
    # Upload the file
    s3_client.upload_fileobj(buffer, input_data['output_bucket'], input_data['output_key'])

    output_path = f"s3://{input_data['output_bucket']}/{input_data['output_key']}"
    print(f"uploaded the file to {output_path}")

    # Return the butes from the BytesIO object
    return output_path


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

    print('Received input: %s', request_body)
    
    if request_content_type == "application/json":
        try:
            request = json.loads(request_body)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format in request body")
    else:
        raise ValueError("Unsupported content type: {}".format(request_content_type))


    # Extract and validate required fields
    required_fields = ["text", "output_bucket", "output_key"]
    missing_fields = [field for field in required_fields if field not in request]
    if missing_fields:
        raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")


    # Extract and handle optional fields with defaults
    return {
        "text": request.get("text", ""),
        "voice_id": request.get("voice_id", "male_voice"),
        "output_bucket": request.get("output_bucket"),
        "output_key": request.get("output_key")
    }

def output_fn(response_body, response_content_type):

    """
    Serialize and prepare the prediction output
    """

    print('Returning response ===============')
    return {
        "statusCode": 200,
        "body": response_body}