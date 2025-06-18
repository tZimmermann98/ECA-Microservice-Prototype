import streamlit as st
import requests
import os
import time
import boto3
from botocore.exceptions import ClientError
import io

from utils.db import SessionLocal, User, Avatar, Session as DbSession, Interaction

ORCHESTRATION_SERVICE_URL = os.getenv("ORCHESTRATION_SERVICE_URL", "http://orchestration-service:8000")

S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "http://minio:9000")
S3_ACCESS_KEY_ID = os.getenv("S3_ACCESS_KEY_ID")
S3_SECRET_ACCESS_KEY = os.getenv("S3_SECRET_ACCESS_KEY")
S3_INPUT_BUCKET_NAME = os.getenv("S3_INPUT_BUCKET_NAME", "inputs")
S3_VIDEO_OUTPUT_BUCKET_NAME = os.getenv("S3_VIDEO_OUTPUT_BUCKET_NAME", "video-outputs")

POLLING_INTERVAL = 10  # seconds
POLLING_TIMEOUT = 3600 # 1 hour


def get_user(db, email):
    return db.query(User).filter(User.user_email == email).first()

def create_user(db, email, name):
    user = get_user(db, email)
    if user:
        user.user_name = name
    else:
        user = User(user_email=email, user_name=name)
        db.add(user)
    db.commit()
    db.refresh(user)
    return user

def get_avatars(db):
    return db.query(Avatar).all()

def get_user_sessions(db, user_id, avatar_id):
    return db.query(DbSession).filter(DbSession.user_id == user_id, DbSession.avatar_id == avatar_id).order_by(DbSession.start_time.desc()).all()

def create_new_session(db, user_id, avatar_id):
    new_session = DbSession(user_id=user_id, avatar_id=avatar_id)
    db.add(new_session)
    db.commit()
    db.refresh(new_session)
    return new_session

def get_session_history(db, session_id):
    return db.query(Interaction).filter(Interaction.session_id == session_id).order_by(Interaction.timestamp.asc()).all()

def upload_to_s3(file_data, file_name):
    s3_client = boto3.client('s3', endpoint_url=S3_ENDPOINT_URL, aws_access_key_id=S3_ACCESS_KEY_ID, aws_secret_access_key=S3_SECRET_ACCESS_KEY)
    try:
        s3_key = f"user_uploads/{int(time.time())}_{file_name}"
        s3_client.upload_fileobj(file_data, S3_INPUT_BUCKET_NAME, s3_key)
        return s3_key
    except ClientError as e:
        st.error(f"Failed to upload file to S3: {e}")
        return None

def get_s3_object_bytes(bucket_name, key):
    s3_client = boto3.client('s3', endpoint_url=S3_ENDPOINT_URL, aws_access_key_id=S3_ACCESS_KEY_ID, aws_secret_access_key=S3_SECRET_ACCESS_KEY)
    try:
        byte_buffer = io.BytesIO()
        s3_client.download_fileobj(bucket_name, key, byte_buffer)
        return byte_buffer.getvalue()
    except ClientError as e:
        st.error(f"Failed to retrieve file '{key}' from S3: {e}")
        return None

st.set_page_config(page_title="Embodied Conversational Agent", layout="wide")
st.title("Embodied Conversational Agent")

if "X-Forwarded-Email" not in st.context.headers:
    st.error("Authentication Error: User email not found. Please log in.")
    st.stop()

email = st.context.headers["X-Forwarded-Email"]
db = SessionLocal()

# 2. User Registration Check
user = get_user(db, email)
if not user or not user.user_name:
    st.info("Welcome! Please enter your name to get started.")
    with st.form("name_form"):
        name = st.text_input("Your Name", value=(user.user_name if user else ""))
        if st.form_submit_button("Save"):
            user = create_user(db, email, name)
            st.rerun()
    st.stop()

st.success(f"Welcome, {user.user_name} ({email})!")

with st.sidebar:
    st.header("Configuration")
    
    avatars = get_avatars(db)
    if not avatars:
        st.warning("No avatars configured in the database.")
        st.stop()
        
    avatar_options = {avatar.avatar_name: avatar.avatar_id for avatar in avatars}
    selected_avatar_name = st.selectbox("Choose an Avatar", options=avatar_options.keys())
    st.session_state.selected_avatar_id = avatar_options[selected_avatar_name]

    if st.button("✨ New Session"):
        new_session = create_new_session(db, user.user_id, st.session_state.selected_avatar_id)
        st.session_state.selected_session_id = new_session.session_id
        st.rerun()

    sessions = get_user_sessions(db, user.user_id, st.session_state.selected_avatar_id)
    session_options = {f"Session {s.session_id}": s.session_id for s in sessions}
    
    if session_options:
        st.session_state.selected_session_id = st.selectbox("Choose a Session", options=session_options.values(), format_func=lambda x: f"Session {x}")
    else:
        st.write("No sessions for this avatar. Start a new one!")
        st.session_state.selected_session_id = None

if "selected_session_id" in st.session_state and st.session_state.selected_session_id:
    
    history = get_session_history(db, st.session_state.selected_session_id)
    for interaction in history:
        with st.chat_message("user"):
            if interaction.user_input_video_url:
                video_bytes = get_s3_object_bytes(S3_INPUT_BUCKET_NAME, interaction.user_input_video_url)
                if video_bytes: st.video(video_bytes)
            elif interaction.user_input_audio_url and not interaction.user_input_video_url: # Only show audio if no video
                audio_bytes = get_s3_object_bytes(S3_INPUT_BUCKET_NAME, interaction.user_input_audio_url)
                if audio_bytes: st.audio(audio_bytes)
            elif interaction.user_input_text:
                st.markdown(interaction.user_input_text)
        
        with st.chat_message("assistant"):
            if interaction.status == "completed":
                if interaction.generated_video_url:
                    video_bytes = get_s3_object_bytes(S3_VIDEO_OUTPUT_BUCKET_NAME, interaction.generated_video_url)
                    if video_bytes: st.video(video_bytes)
                else:
                    st.markdown(interaction.agent_response_text or "Sorry, I couldn't generate a video response.")
            
            elif interaction.status == "processing":
                with st.spinner("Avatar is preparing a response... This may take a few minutes."):
                    max_retries = POLLING_TIMEOUT // POLLING_INTERVAL
                    for i in range(max_retries):
                        time.sleep(POLLING_INTERVAL)
                        db.refresh(interaction) # Get latest status from DB
                        if interaction.status != "processing":
                            st.rerun() # Rerun to display the final state
                    # If the loop finishes, it timed out
                    st.error("The request timed out. Please try again.")
            
            else: # Handle failed states
                st.error(f"Sorry, the response failed. Status: {interaction.status}")

    with st.form("chat-form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])
        with col1:
            prompt = st.text_area("Your message:", key="text_input")
            uploaded_file = st.file_uploader("Upload a file (video or audio)", type=["mp4", "mov", "wav", "mp3"])
        with col2:
            submitted = st.form_submit_button("Send")

    if submitted and (prompt or uploaded_file):
        input_data = {"session_id": st.session_state.selected_session_id}
        if uploaded_file:
            s3_key = upload_to_s3(uploaded_file, uploaded_file.name)
            if "video" in uploaded_file.type:
                input_data["input_video_s3_key"] = s3_key
            else:
                input_data["input_audio_s3_key"] = s3_key
        if prompt:
            input_data["input_text"] = prompt
        
        try:
            response = requests.post(f"{ORCHESTRATION_SERVICE_URL}/v1/orchestrate", json=input_data)
            response.raise_for_status()
            st.rerun()
        except requests.exceptions.RequestException as e:
            st.error(f"Could not connect to the backend service: {e}")
            
else:
    st.info("Please select or create a session in the sidebar to begin.")

db.close()
