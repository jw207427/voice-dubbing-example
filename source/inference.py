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

def load_autogressive_model(model, model_id):
    """
    Given a model_id, load the autogressive model weights from the model directory
    
    Args:
        model (Tortoise): The Tortoise model to load the weights into.
        model_id (str): The identifier for the model to load.
            Currently supported model_id include female_english, male_german, male_spanish
    """

    logger.info("Loading model weights")

    # Load the model weights
    if model_id == 'female_english':
        logger.info("Loading female english model")
        model_path = os.path.join(MODEL_DIR, 'female_english/autoregressive.pth')
    elif model_id == 'male_german':
        logger.info("Loading male german model")
        model_path = os.path.join(MODEL_DIR, 'male_german/autoregressive.pth')
    elif model_id == 'male_spanish':
        logger.info("Loading male spanish model")
        model_path = os.path.join(MODEL_DIR, 'male_spanish/autoregressive.pth')
    elif model_id == 'male_japanese':
        logger.info("Loading male japanese model")
        model_path = os.path.join(MODEL_DIR, 'male_japanese/autoregressive.pth')
    
    model.autoregressive = UnifiedVoice(max_mel_tokens=604, max_text_tokens=402, max_conditioning_inputs=2, layers=30,
                                          model_dim=1024,
                                          heads=16, number_text_tokens=255, start_text_token=255, checkpointing=False,
                                          train_solo_embeddings=False).cpu().eval()

    model.autoregressive.load_state_dict(torch.load(model_path), strict=False)
    model.autoregressive.post_init_gpt2_config(use_deepspeed=USE_DEEPSPEED, kv_cache=KV_CACHE, half=HALF)
    logger.info("Model loaded")


def download_get_conditioning_latents(model, voice_samples_s3_uri):
    """
    Download voice samples from S3 and compute conditioning latents

    Args:
        model (Tortoise): The Tortoise model to load the weights into.
        voice_samples_s3_uri (str): The S3 URI for the voice samples.
    """

    logger.info("Downloading voice samples")


    # Download all the voice samples (.wav) files from S3 into a temporary folder
    # for each object in the s3 uri, download it
    session = boto3.Session()
    s3_client = session.client('s3')
    
    
    uri = voice_samples_s3_uri
    parts = uri.split('//')[1].split('/')
    bucket_name = parts[0]
    key = '/'.join(parts[1:])  # Reconstruct key with leading slash

    print("Bucket name:", bucket_name)
    print("Key:", key)
        
    
    logger.info(f"Source Bucket: {bucket_name}, Source Key: {key}")
    # Use a temporary directory and download all files
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Created temporary directory {tmpdir}")
        
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=key)
        for object in response['Contents']:
            #skip directories
            if object['Key'].endswith('/'):
                continue
            #download the s3 object
            s3_client.download_file(bucket_name, object['Key'], os.path.join(tmpdir,object['Key'].split('/')[-1]))
        # Load all downloaded files as a single tensor stack
        voice_samples = [audio.load_audio(os.path.join(tmpdir, filename), 22050) for filename in os.listdir(tmpdir)]

    logger.info("Computing conditioning latents")
    # Compute conditioning latentstents for the given voice samples.
    conditioning_latents = model.get_conditioning_latents(voice_samples)
    logger.info("Conditioning latents computed")
    
    return conditioning_latents
    
def predict_fn(input_data, model):
    """
    Run prediction on input data
    """
    
    if input_data['model_id']:
        load_autogressive_model(model, input_data['model_id'])
    
    if input_data['voice_samples_s3_uri']:
        conditioning_latents = download_get_conditioning_latents(model, input_data['voice_samples_s3_uri'])
    
    logger.info("Generating with params: %s", input_data)

    # Synthesize
    if input_data['voice_samples_s3_uri']:
        audio_clip = model.tts(input_data['text'], voice_samples=None, conditioning_latents=conditioning_latents).squeeze(0).cpu()
    else:
        audio_clip = model.tts_with_preset(input_data['text'], 
                                           preset="fast").squeeze(0).cpu()

    #Save to buffer
    
    # Generate unique filename
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    filename = f"output-{input_data['model_id']}-{timestamp}.wav"

    # Save to temporary file
    torchaudio.save(os.path.join('/tmp', filename), audio_clip, 24000, format="wav")
    
    s3_uri = input_data['destination_s3_uri']
    parts = s3_uri.split('//')[1].split('/')
    bucket_name = parts[0]
    key = '/'.join(parts[1:]) 
    
    # Configure boto3 client
    session = boto3.Session()
    s3_client = session.client('s3')

    # Upload file to S3
    s3_client.upload_file(os.path.join('/tmp', filename), bucket_name, key)

    # Optionally, delete the temporary file after upload
    os.remove(os.path.join('/tmp', filename))

   # Construct complete S3 URI
    complete_s3_uri = f"s3://{bucket_name}/{key}"

    print(f"file saved to s3: {complete_s3_uri}")

    # Return complete S3 URI on success
    return complete_s3_uri


def input_fn(request_body, request_content_type):
    """
    Deserializes the input request body and prepares data for text-to-speech generation.

    Args:
        request_body (str): A JSON string containing the following fields:

            text (str): The text to be synthesized into speech.
            voice_samples_s3_uri (str): The S3 URI pointing to a prefix containing voice samples for personalization.
            destination_s3_uri (str): The S3 URI where the generated voice output will be stored.
            model_id (str): Identifier for the autoregressive model to be used for synthesis.
            inference_params (dict): A dictionary containing parameters controlling the tortoise-tts generation process.

    Returns:
        A dict containing:
            text (str): The extracted text to be generated.
            voice_samples_s3_uri (str): The S3 URI for voice samples (if provided).
            destination_s3_uri (str): The S3 URI for generated audio output.
            model_id (str): The identified model to use (default: None).
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
    required_fields = ["text", "voice_samples_s3_uri", "destination_s3_uri", "model_id"]
    missing_fields = [field for field in required_fields if field not in request]
    if missing_fields:
        raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

    # Extract and handle optional fields with defaults
    return {
        "text": request["text"],
        "voice_samples_s3_uri": request.get("voice_samples_s3_uri", None),
        "destination_s3_uri": request.get("destination_s3_uri"),
        "model_id": request.get("model_id"),
        "inference_params": request.get("inference_params", {}),
    }


def output_fn(response_body, response_content_type):

    """
    Serialize and prepare the prediction output
    """

    logger.info('Returning response')
    return {
        "statusCode": 200,
        "output_s3_uri": response_body}