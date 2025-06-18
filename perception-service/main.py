import os
import tempfile
import uvicorn
import cv2
import logging
import numpy as np
import boto3
from collections import Counter
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from moviepy import VideoFileClip
from sqlalchemy.orm import Session

from utils.db import get_db, Interaction

try:
    from deepface import DeepFace
    print("DeepFace library imported successfully.")
    try:
        print("Pre-loading DeepFace attribute models...")
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        DeepFace.analyze(dummy_image, actions=['emotion', 'age', 'gender'], enforce_detection=False)
        print("DeepFace models pre-loaded successfully.")
    except Exception as e:
        print(f"Could not pre-load DeepFace models: {e}.")
except ImportError:
    print("DeepFace not found.")
    DeepFace = None

# SpeechBrain and Whisper model loading
asr_model = None
audio_emotion_classifier = None
try:
    from speechbrain.inference.interfaces import foreign_class
    import whisper
    print("SpeechBrain and Whisper libraries imported successfully.")
    try:
        AUDIO_EMOTION_MODEL = os.getenv("AUDIO_EMOTION_MODEL", "speechbrain/emotion-recognition-wav2vec2-IEMOCAP")
        audio_emotion_classifier = foreign_class(source=AUDIO_EMOTION_MODEL, pymodule_file="custom_interface.py", classname="CustomEncoderWav2vec2Classifier")
        print("SpeechBrain Emotion classifier loaded.")
    except Exception as e:
        print(f"ERROR: Could not load SpeechBrain Emotion model: {e}")
    try:
        WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "base")
        asr_model = whisper.load_model(WHISPER_MODEL_NAME)
        print(f"Whisper ASR model ({WHISPER_MODEL_NAME}) loaded.")
    except Exception as e:
        print(f"ERROR: Could not load Whisper model: {e}")
except ImportError:
    print("Could not import SpeechBrain or Whisper.")

S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL")
S3_ACCESS_KEY_ID = os.getenv("S3_ACCESS_KEY_ID")
S3_SECRET_ACCESS_KEY = os.getenv("S3_SECRET_ACCESS_KEY")
S3_INPUT_BUCKET_NAME = os.getenv("S3_INPUT_BUCKET_NAME", "inputs")

app = FastAPI(
    title="Perception Engine",
    description="Analyzes files from S3 based on an interaction ID and updates the database.",
    version="1.0.0"
)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerceptionRequest(BaseModel):
    interaction_id: int = Field(..., description="The ID of the interaction to be processed.")

class PerceptionResponse(BaseModel):
    interaction_id: int
    status: str = "perception_complete"
    transcribed_text: str | None
    perceived_user_affect: str | None

def get_s3_client():
    return boto3.client('s3', endpoint_url=S3_ENDPOINT_URL, aws_access_key_id=S3_ACCESS_KEY_ID, aws_secret_access_key=S3_SECRET_ACCESS_KEY)

def extract_audio_from_video(video_path: str) -> str:
    try:
        video_clip = VideoFileClip(video_path)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
            audio_path = temp_audio_file.name
        video_clip.audio.write_audiofile(audio_path, codec='pcm_s16le', fps=16000)
        video_clip.close()
        return audio_path
    except Exception as e:
        logger.error(f"Failed to extract audio from video: {e}")
        return None

def analyze_video_frames(video_path: str) -> dict:
    if not DeepFace: return {}
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return {}
    frame_rate = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = int(frame_rate)
    all_emotions, all_ages, all_genders = [], [], []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if frame_count % frame_interval == 0:
            try:
                analysis = DeepFace.analyze(frame, actions=['emotion', 'age', 'gender'], enforce_detection=False)
                if isinstance(analysis, list) and analysis:
                    face_data = analysis[0]
                    all_emotions.append(face_data.get('dominant_emotion'))
                    all_ages.append(face_data.get('age'))
                    all_genders.append(face_data.get('dominant_gender'))
            except Exception: pass
        frame_count += 1
    cap.release()
    if not all_emotions: return {}
    return {
        "video_emotion": Counter(all_emotions).most_common(1)[0][0] if all_emotions else None,
        "age": int(sum(all_ages) / len(all_ages)) if all_ages else None,
        "gender": Counter(all_genders).most_common(1)[0][0] if all_genders else None
    }

def analyze_audio_emotion(file_path: str) -> str:
    if not audio_emotion_classifier: return "Unknown"
    try:
        _, _, _, text_lab = audio_emotion_classifier.classify_file(file_path)
        emotion = text_lab[0] if isinstance(text_lab, list) else text_lab
        emotion_map = {'neu': 'Neutral', 'ang': 'Angry', 'hap': 'Happy', 'sad': 'Sadness'}
        return emotion_map.get(emotion.lower(), 'Unknown')
    except Exception: return "Unknown"

def transcribe_audio_with_whisper(file_path: str) -> str:
    if not asr_model: raise HTTPException(status_code=501, detail="Whisper ASR model not loaded.")
    try:
        result = asr_model.transcribe(file_path)
        return result.get("text", "")
    except Exception: return ""

@app.post("/v1/analyze", response_model=PerceptionResponse)
async def analyze_input(request: PerceptionRequest, db: Session = Depends(get_db)):
    """
    Fetches file keys from the database based on an interaction_id,
    performs analysis, and updates the database record.
    """
    logger.info(f"Received request to analyze interaction_id: {request.interaction_id}")
    
    interaction = db.query(Interaction).filter(Interaction.interaction_id == request.interaction_id).first()
    if not interaction:
        raise HTTPException(status_code=404, detail="Interaction not found.")

    video_path, audio_path, extracted_audio_path = None, None, None
    transcribed_text, affect_parts = None, []
    
    s3_client = get_s3_client()
    if not s3_client: raise HTTPException(status_code=500, detail="S3 client not available.")

    try:
        if interaction.user_input_video_url:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                s3_client.download_fileobj(S3_INPUT_BUCKET_NAME, interaction.user_input_video_url, temp_file)
                video_path = temp_file.name
            video_analysis = analyze_video_frames(video_path)
            if video_analysis:
                if video_analysis.get("video_emotion"): affect_parts.append(f"Visually, the user appears {video_analysis['video_emotion']}.")
                if video_analysis.get("age"): affect_parts.append(f"They seem to be around {video_analysis['age']} years old.")
                if video_analysis.get("gender"): affect_parts.append(f"Their perceived gender is {video_analysis['gender']}.")
            extracted_audio_path = extract_audio_from_video(video_path)
            audio_path = extracted_audio_path
        
        elif interaction.user_input_audio_url:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                s3_client.download_fileobj(S3_INPUT_BUCKET_NAME, interaction.user_input_audio_url, temp_file)
                audio_path = temp_file.name

        if audio_path:
            audio_emotion = analyze_audio_emotion(audio_path)
            if audio_emotion != "Unknown":
                affect_parts.append(f"Vocally, their tone sounds {audio_emotion}.")
            transcribed_text = transcribe_audio_with_whisper(audio_path)
    
    finally:
        if video_path: os.remove(video_path)
        if audio_path: os.remove(audio_path)

    # Update the interaction record in the database
    interaction.user_input_text = transcribed_text
    interaction.perceived_user_affect = " ".join(affect_parts) if affect_parts else None
    db.commit()
    logger.info(f"Updated interaction {request.interaction_id} with perception data.")

    return PerceptionResponse(
        interaction_id=request.interaction_id,
        transcribed_text=transcribed_text,
        perceived_user_affect=interaction.perceived_user_affect
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
