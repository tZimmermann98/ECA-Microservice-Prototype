import os
import uvicorn
import logging
import httpx
import boto3
import asyncio
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from typing import List, Literal
from sqlalchemy.orm import Session, joinedload
from openai import OpenAI

from utils.db import get_db, Interaction, UserMemory, AvatarMemory

PERCEPTION_SERVICE_URL = os.getenv("PERCEPTION_SERVICE_URL", "http://perception-service:8000")
CONVERSATIONAL_SERVICE_URL = os.getenv("CONVERSATIONAL_SERVICE_URL", "http://conversational-service:8000")
VOCAL_SERVICE_URL = os.getenv("VOCAL_SERVICE_URL", "http://vocal-service:8000")
EMBODIMENT_SERVICE_URL = os.getenv("EMBODIMENT_SERVICE_URL", "http://embodiment-service:8004")

LLM_HOST = os.getenv("LLM_HOST", "https://api.openai.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")

S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL")
S3_ACCESS_KEY_ID = os.getenv("S3_ACCESS_KEY_ID")
S3_SECRET_ACCESS_KEY = os.getenv("S3_SECRET_ACCESS_KEY")
S3_INPUT_BUCKET_NAME = os.getenv("S3_INPUT_BUCKET_NAME", "inputs")

POLLING_INTERVAL = 10 # seconds
POLLING_TIMEOUT = 3600 # 1 hour

app = FastAPI(
    title="Orchestration Engine",
    description="Manages the end-to-end workflow for generating an avatar response.",
    version="1.0.0"
)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OrchestrationRequest(BaseModel):
    session_id: int
    input_text: str | None = None
    input_audio_s3_key: str | None = None
    input_video_s3_key: str | None = None

class OrchestrationResponse(BaseModel):
    interaction_id: int
    message: str = "Orchestration process started."

class MemoryItem(BaseModel):
    memory_kind: Literal["user", "avatar"]
    memory_key: str
    memory_value: str

class MemoryExtractionTool(BaseModel):
    memories: List[MemoryItem]

def get_s3_client():
    return boto3.client('s3', endpoint_url=S3_ENDPOINT_URL, aws_access_key_id=S3_ACCESS_KEY_ID, aws_secret_access_key=S3_SECRET_ACCESS_KEY)

def construct_memory_extraction_prompt(user_input: str, avatar_response: str) -> str:
    """Creates the detailed prompt for the memory extraction LLM."""
    return f"""You are a memory extraction agent. Your task is to analyze a conversation turn and determine if any new, lasting information about the USER or the AVATAR should be saved.

Analyze the following conversation:
User Input: "{user_input}"
Avatar Response: "{avatar_response}"

Based on this exchange, should any new facts be stored?
- If the user reveals personal information (e.g., "My name is Tobias", "I'm interested in DSR"), extract it for USER memory.
- If the avatar makes a statement about itself or its capabilities (e.g., "You can call me Jan"), extract it for AVATAR memory.
- If no new, lasting information is revealed, call the tool with an empty list.

Provide your answer ONLY by calling the 'save_memories' tool with a valid list of memory objects.
"""

async def run_orchestration_pipeline(interaction_id: int):
    """The main orchestration logic that runs in the background."""
    db = get_db() 
    try:
        # Step 1: Fetch the newly created interaction
        interaction = db.query(Interaction).options(joinedload(Interaction.session)).filter(Interaction.interaction_id == interaction_id).one()
        session = interaction.session
        user_id = session.user_id
        avatar_id = session.avatar_id

        # Step 2: Perception (if needed)
        if interaction.user_input_audio_url or interaction.user_input_video_url:
            logger.info(f"[Interaction {interaction_id}] Calling Perception Service...")
            
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    f"{PERCEPTION_SERVICE_URL}/v1/analyze", 
                    json={"interaction_id": interaction.interaction_id}
                )
                response.raise_for_status()
                logger.info(f"[Interaction {interaction_id}] Perception Service response: {response.text}")
            
            db.refresh(interaction) 
            if not interaction.user_input_text:
                logger.error(f"[Interaction {interaction_id}] No text transcribed from input. Stopping orchestration.")
                interaction.status = "failed_no_transcription"
                db.commit()
                return # End the background task

        # Step 3: Conversational
        logger.info(f"[Interaction {interaction_id}] Calling Conversational Service...")
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(f"{CONVERSATIONAL_SERVICE_URL}/v1/generate-response", json={"interaction_id": interaction.interaction_id})
            response.raise_for_status()
        conversation_data = response.json()
        
        interaction.raw_content_response = conversation_data.get("raw_content_response")
        interaction.agent_response_text = conversation_data.get("final_response_text")
        db.commit()
        logger.info(f"[Interaction {interaction_id}] Conversation complete. Response: '{interaction.agent_response_text}'")

        # Step 4: Memory Extraction
        logger.info(f"[Interaction {interaction_id}] Starting Memory Extraction...")
        try:
            memory_client = OpenAI(base_url=LLM_HOST, api_key=LLM_API_KEY)
            prompt = construct_memory_extraction_prompt(interaction.user_input_text, interaction.agent_response_text)
            
            messages = [{"role": "system", "content": "You are a memory extraction agent."}, {"role": "user", "content": prompt}]
            
            response = memory_client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                tools=[{"type": "function", "function": {"name": "save_memories", "description": "Saves a list of new memories.", "parameters": MemoryExtractionTool.model_json_schema()}}],
                tool_choice={"type": "function", "function": {"name": "save_memories"}}
            )
            
            tool_call = response.choices[0].message.tool_calls[0] if response.choices[0].message.tool_calls else None
            if tool_call and tool_call.function.name == "save_memories":
                extracted_data = MemoryExtractionTool.model_validate_json(tool_call.function.arguments)
                for item in extracted_data.memories:
                    if item.memory_kind == "user":
                        db.add(UserMemory(user_id=user_id, memory_key=item.memory_key, memory_value=item.memory_value))
                    elif item.memory_kind == "avatar":
                        db.add(AvatarMemory(avatar_id=avatar_id, memory_key=item.memory_key, memory_value=item.memory_value))
                db.commit()
                logger.info(f"Saved {len(extracted_data.memories)} new memories.")
            else:
                logger.info("No new memories to save.")
        except Exception as e:
            logger.error(f"Memory extraction failed: {e}")
            db.rollback()

        # Step 5: Vocal
        logger.info(f"[Interaction {interaction_id}] Calling Vocal Service...")
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(f"{VOCAL_SERVICE_URL}/v1/synthesize", json={"interaction_id": interaction.interaction_id})
            response.raise_for_status()
        
        # Step 6: Embodiment
        logger.info(f"[Interaction {interaction_id}] Calling Embodiment Service...")
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(f"{EMBODIMENT_SERVICE_URL}/v1/generate", json={"interaction_id": interaction.interaction_id})
            response.raise_for_status()

        # Step 7: Poll for Video Completion
        logger.info(f"[Interaction {interaction_id}] Started polling for video completion...")
        max_attempts = POLLING_TIMEOUT // POLLING_INTERVAL
        for attempt in range(max_attempts):
            await asyncio.sleep(POLLING_INTERVAL)
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.get(f"{EMBODIMENT_SERVICE_URL}/v1/status/{interaction_id}")
                    response.raise_for_status()
                
                status_data = response.json()
                status = status_data.get("status")
                logger.info(f"Polling attempt {attempt + 1}/{max_attempts}: Status is '{status}'")

                if status == "completed":
                    interaction.status = "completed"
                    db.commit()
                    logger.info(f"--- Orchestration for Interaction {interaction_id} complete! ---")
                    return # Exit the background task successfully
                elif status == "failed":
                    interaction.status = "failed"
                    db.commit()
                    raise Exception("Video generation failed according to status endpoint.")

            except Exception as e:
                interaction.status = "failed"
                db.commit()
                logger.error(f"Error during polling: {e}")
        
        # If loop finishes without completion
        raise Exception("Polling timed out.")

    except Exception as e:
        logger.error(f"--- Orchestration FAILED for Interaction {interaction_id}: {e} ---")
        try:
            interaction.status = "failed_orchestration"
            db.commit()
        except Exception as db_e:
            logger.error(f"Failed to even mark interaction as failed: {db_e}")
            db.rollback()

@app.post("/v1/orchestrate", response_model=OrchestrationResponse)
async def orchestrate_interaction(request: OrchestrationRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """
    Receives an initial request, creates an Interaction record, and
    triggers the full generation pipeline in the background.
    """
    logger.info(f"Received orchestration request for session_id: {request.session_id}")

    try:
        new_interaction = Interaction(
            session_id=request.session_id,
            user_input_text=request.input_text,
            user_input_audio_url=request.input_audio_s3_key,
            user_input_video_url=request.input_video_s3_key,
            status="processing"
        )
        db.add(new_interaction)
        db.commit()
        db.refresh(new_interaction)
        
        background_tasks.add_task(run_orchestration_pipeline, new_interaction.interaction_id)

        return OrchestrationResponse(interaction_id=new_interaction.interaction_id)

    except Exception as e:
        logger.error(f"Failed to create initial interaction record: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Could not initiate orchestration.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
