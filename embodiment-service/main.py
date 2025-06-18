import os
import uvicorn
import logging
import httpx
import boto3
import tempfile
from botocore.exceptions import ClientError
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session, joinedload

from utils.db import get_db, Avatar, Interaction, Session as DbSession

S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL")
S3_ACCESS_KEY_ID = os.getenv("S3_ACCESS_KEY_ID")
S3_SECRET_ACCESS_KEY = os.getenv("S3_SECRET_ACCESS_KEY")
S3_VIDEO_BUCKET_NAME = os.getenv("S3_VIDEO_BUCKET_NAME", "video-outputs")
S3_AUDIO_BUCKET_NAME = os.getenv("S3_AUDIO_BUCKET_NAME", "audio-outputs")

app = FastAPI(
    title="Embodiment Engine",
    description="Starts and monitors the generation of a talking head video.",
    version="1.0.0"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbodimentRequest(BaseModel):
    interaction_id: int = Field(..., description="The ID of the interaction to embody.")

class EmbodimentResponse(BaseModel):
    heygen_video_id: str = Field(..., description="The video ID returned by HeyGen for tracking.")
    message: str = "Video generation started successfully."

class StatusResponse(BaseModel):
    status: str = Field(..., description="The current status of the video generation (e.g., 'processing', 'completed', 'failed').")
    generated_video_key: str | None = Field(None, description="The S3 key for the final video if completed.")

def get_s3_client():
    try:
        s3_client = boto3.client('s3', endpoint_url=S3_ENDPOINT_URL, aws_access_key_id=S3_ACCESS_KEY_ID, aws_secret_access_key=S3_SECRET_ACCESS_KEY)
        return s3_client
    except Exception as e:
        logger.error(f"Failed to create S3 client: {e}")
        return None

@app.post("/v1/generate", response_model=EmbodimentResponse)
async def generate_video(request: EmbodimentRequest, db: Session = Depends(get_db)):
    """Starts the video generation process and returns a task ID."""
    logger.info(f"Received request to generate video for interaction_id: {request.interaction_id}")

    # 1. Fetch data
    interaction = db.query(Interaction).options(joinedload(Interaction.session).joinedload(DbSession.avatar).joinedload(Avatar.video_provider)).filter(Interaction.interaction_id == request.interaction_id).first()
    if not interaction or not interaction.session.avatar.video_provider:
        raise HTTPException(status_code=404, detail="Interaction or associated video provider not found.")
    video_provider = interaction.session.avatar.video_provider
    audio_s3_key = interaction.generated_audio_url
    if not audio_s3_key:
        raise HTTPException(status_code=404, detail="No generated audio key found.")

    # 2. Download Audio from S3
    s3_client = get_s3_client()
    if not s3_client: raise HTTPException(status_code=500, detail="S3 client not available.")
    temp_audio_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            s3_client.download_fileobj(S3_AUDIO_BUCKET_NAME, audio_s3_key, temp_file)
            temp_audio_path = temp_file.name
    except ClientError as e:
        raise HTTPException(status_code=404, detail=f"Audio file not found in S3: {e}")
    
    # 3. Upload Audio Asset to HeyGen
    api_key = os.getenv(video_provider.api_key_env_var)
    if not api_key: raise HTTPException(status_code=500, detail="HeyGen API key not configured.")
    audio_asset_id = None
    try:
        with open(temp_audio_path, "rb") as f:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post("https://upload.heygen.com/v1/asset", headers={"x-api-key": api_key, "Content-Type": "audio/mpeg"}, data=f.read())
                response.raise_for_status()
                audio_asset_id = response.json().get("data", {}).get("id")
        if not audio_asset_id: raise ValueError("Could not extract audio asset ID.")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Error uploading asset to HeyGen: {e}")
    finally:
        if temp_audio_path: os.remove(temp_audio_path)

    # 4. Start Video Generation
    payload = {"video_inputs": [{"character": {"type": "talking_photo", "talking_photo_id": video_provider.provider_avatar_id}, "voice": {"type": "audio", "audio_asset_id": audio_asset_id}}], "dimension": {"width": 1280, "height": 720}}
    heygen_video_id = None
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post("https://api.heygen.com/v2/video/generate", headers={"x-api-key": api_key, "Content-Type": "application/json"}, json=payload)
            response.raise_for_status()
            heygen_video_id = response.json().get("data", {}).get("video_id")
        if not heygen_video_id: raise ValueError("Could not extract video_id.")
        
        # 5. Store the provider's task ID in our database
        interaction.video_provider_task_id = heygen_video_id
        db.commit()
        
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Error starting HeyGen video generation: {e}")

    return EmbodimentResponse(heygen_video_id=heygen_video_id)

@app.get("/v1/status/{interaction_id}", response_model=StatusResponse)
async def get_video_status(interaction_id: int, db: Session = Depends(get_db)):
    """
    Checks the status of a video generation task. If complete, downloads the
    video and saves it to S3.
    """
    logger.info(f"Checking status for interaction_id: {interaction_id}")
    
    # 1. Fetch the interaction record to get the provider task ID
    interaction = db.query(Interaction).options(joinedload(Interaction.session).joinedload(DbSession.avatar).joinedload(Avatar.video_provider)).filter(Interaction.interaction_id == interaction_id).first()
    if not interaction or not interaction.video_provider_task_id:
        raise HTTPException(status_code=404, detail="Interaction or provider task ID not found.")
    
    # 2. Call HeyGen status endpoint
    video_provider = interaction.session.avatar.video_provider
    api_key = os.getenv(video_provider.api_key_env_var)
    status_url = f"https://api.heygen.com/v1/video_status.get?video_id={interaction.video_provider_task_id}"
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(status_url, headers={"x-api-key": api_key})
            response.raise_for_status()
            status_data = response.json().get("data", {})
        video_status = status_data.get("status")
        logger.info(f"HeyGen status for video {interaction.video_provider_task_id} is: {video_status}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Error checking video status with HeyGen: {e}")

    # 3. Process the status
    if video_status == "completed":
        final_video_url = status_data.get("video_url")
        if not final_video_url:
            raise HTTPException(status_code=500, detail="Video completed but no URL provided by HeyGen.")
        
        # 3a. Download the final video
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                video_response = await client.get(final_video_url)
                video_response.raise_for_status()
                video_content = video_response.content
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Failed to download completed video from provider: {e}")
            
        # 3b. Upload final video to our S3
        s3_client = get_s3_client()
        if not s3_client: raise HTTPException(status_code=500, detail="S3 client not available.")
        video_s3_key = f"interaction_{interaction_id}.mp4"
        try:
            s3_client.put_object(Bucket=S3_VIDEO_BUCKET_NAME, Key=video_s3_key, Body=video_content, ContentType='video/mp4')
            
            # 3c. Update our database record
            interaction.generated_video_url = video_s3_key
            db.commit()
            
            return StatusResponse(status="completed", generated_video_key=video_s3_key)
        except Exception as e:
            db.rollback()
            raise HTTPException(status_code=500, detail=f"Failed to upload final video to S3 or update DB: {e}")
    
    elif video_status == "failed":
        return StatusResponse(status="failed", generated_video_key=None)
    else: # Still processing
        return StatusResponse(status=video_status, generated_video_key=None)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
