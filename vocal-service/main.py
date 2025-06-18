import os
import uvicorn
import logging
import httpx
import boto3
from botocore.exceptions import ClientError
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session, joinedload

from utils.db import get_db, Avatar, Interaction, Session as DbSession

S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL")
S3_ACCESS_KEY_ID = os.getenv("S3_ACCESS_KEY_ID")
S3_SECRET_ACCESS_KEY = os.getenv("S3_SECRET_ACCESS_KEY")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "audio-outputs")
S3_REGION = os.getenv("S3_REGION", "us-east-1")

app = FastAPI(
    title="Vocal Engine",
    description="Generates audio from text and stores it in an S3-compatible object store.",
    version="1.0.0"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VocalRequest(BaseModel):
    interaction_id: int = Field(..., description="The ID of the interaction to vocalize.")

class VocalResponse(BaseModel):
    interaction_id: int
    generated_audio_key: str = Field(..., description="The S3 key for the generated audio file.")

def get_s3_client():
    """Initializes and returns a boto3 S3 client."""
    try:
        s3_client = boto3.client(
            's3',
            endpoint_url=S3_ENDPOINT_URL,
            aws_access_key_id=S3_ACCESS_KEY_ID,
            aws_secret_access_key=S3_SECRET_ACCESS_KEY,
            region_name=S3_REGION
        )
        return s3_client
    except Exception as e:
        logger.error(f"Failed to create S3 client: {e}")
        return None

@app.post("/v1/synthesize", response_model=VocalResponse)
async def synthesize_speech(request: VocalRequest, db: Session = Depends(get_db)):
    """
    Takes an interaction_id, generates the corresponding audio, and uploads it to S3.
    """
    logger.info(f"Received request to synthesize speech for interaction_id: {request.interaction_id}")

    # 1. Fetch Interaction and related Provider details
    try:
        interaction = db.query(Interaction).options(
            joinedload(Interaction.session).joinedload(DbSession.avatar).joinedload(Avatar.audio_provider)
        ).filter(Interaction.interaction_id == request.interaction_id).first()

        if not interaction or not interaction.session.avatar.audio_provider:
            raise HTTPException(status_code=404, detail="Interaction or associated audio provider not found.")
        
        audio_provider = interaction.session.avatar.audio_provider
        text_to_speak = interaction.agent_response_text
        if not text_to_speak:
            raise HTTPException(status_code=400, detail="No agent_response_text to synthesize.")

    except Exception as e:
        logger.error(f"Database error for interaction {request.interaction_id}: {e}")
        raise HTTPException(status_code=500, detail="Database error.")

    # 2. Call the external TTS provider
    api_key = os.getenv(audio_provider.api_key_env_var) if audio_provider.api_key_env_var else None
    if not api_key:
        raise HTTPException(status_code=500, detail="TTS API key is not configured.")

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"reference_id": audio_provider.provider_voice_id, "text": text_to_speak}

    try:
        logger.info(f"Contacting TTS provider: {audio_provider.provider_endpoint}")
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(audio_provider.provider_endpoint, headers=headers, json=payload)
            response.raise_for_status()
            audio_content = response.content
    except Exception as e:
        logger.error(f"Error calling TTS API: {e}")
        raise HTTPException(status_code=502, detail="Error from external TTS provider.")

    # 3. Upload the generated audio to S3
    s3_client = get_s3_client()
    if not s3_client:
        raise HTTPException(status_code=500, detail="S3 client not available.")

    # Use a .mp3 extension as the provider returns MPEG audio
    s3_object_key = f"interaction_{request.interaction_id}.mp3"
    
    try:
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=s3_object_key,
            Body=audio_content,
            ContentType='audio/mpeg' # Set correct content type for MP3
        )
        logger.info(f"Successfully uploaded audio to S3 bucket '{S3_BUCKET_NAME}' with key '{s3_object_key}'")
    except ClientError as e:
        logger.error(f"Failed to upload audio to S3: {e}")
        raise HTTPException(status_code=500, detail="Could not upload generated audio.")

    # 4. Update the Interaction record in the database
    try:
        interaction.generated_audio_url = s3_object_key
        db.commit()
        logger.info(f"Updated interaction {request.interaction_id} with S3 key.")
    except Exception as e:
        logger.error(f"Failed to update interaction record: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to update database with S3 key.")

    return VocalResponse(
        interaction_id=request.interaction_id,
        generated_audio_key=s3_object_key,
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
