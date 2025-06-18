import os
import uvicorn
import logging
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from openai import OpenAI
from sqlalchemy.orm import Session, joinedload

from utils.db import get_db, Avatar, User, Session as DbSession, Interaction

app = FastAPI(
    title="Conversational Engine",
    description="Generates persona-aligned text responses based on an interaction ID.",
    version="1.0.0"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Pydantic Models for API ---
class ConversationalRequest(BaseModel):
    """Defines the simplified input model, now only requiring the interaction_id."""
    interaction_id: int = Field(..., description="The ID of the interaction to process.")

class ConversationalResponse(BaseModel):
    raw_content_response: str
    final_response_text: str

# --- Helper Function for API Client ---
def get_openai_client(provider_endpoint: str, api_key_env_var: str | None) -> OpenAI:
    api_key = "not-needed"
    if api_key_env_var:
        api_key = os.getenv(api_key_env_var)
        if not api_key:
            logger.error(f"Environment variable '{api_key_env_var}' not set!")
            raise HTTPException(status_code=500, detail="API key configuration error.")
    return OpenAI(base_url=provider_endpoint, api_key=api_key)

# --- API Endpoint ---
@app.post("/v1/generate-response", response_model=ConversationalResponse)
async def generate_response(request: ConversationalRequest, db: Session = Depends(get_db)):
    """Generates a context-aware and persona-aligned textual response using an interaction ID."""
    logger.info(f"Received request for interaction_id: {request.interaction_id}")
    
    # 1. Fetch all context from the database using the interaction_id
    try:
        interaction = db.query(Interaction).options(
            joinedload(Interaction.session).options(
                joinedload(DbSession.user).joinedload(User.memories),
                joinedload(DbSession.avatar).options(
                    joinedload(Avatar.content_provider),
                    joinedload(Avatar.expression_provider),
                    joinedload(Avatar.memories)
                )
            )
        ).filter(Interaction.interaction_id == request.interaction_id).first()

        if not interaction:
            raise HTTPException(status_code=404, detail=f"Interaction with ID {request.interaction_id} not found.")
        
        # Derive all necessary context from the interaction object
        session = interaction.session
        user = session.user
        avatar = session.avatar
        
        if not user or not avatar or not avatar.content_provider or not avatar.expression_provider:
            raise HTTPException(status_code=404, detail="Incomplete configuration for the session's user or avatar.")
        
        # Get the history from the same session, excluding the current interaction
        interaction_history = db.query(Interaction).filter(
            Interaction.session_id == session.session_id,
            Interaction.timestamp < interaction.timestamp
        ).order_by(Interaction.timestamp.asc()).all()

    except Exception as e:
        logger.error(f"Database error while fetching context for interaction {request.interaction_id}: {e}")
        raise HTTPException(status_code=500, detail="Database error.")

    if not interaction.user_input_text:
        raise HTTPException(status_code=400, detail="Interaction has no transcribed text to process.")

    # Format memories and history
    user_memory_str = ", ".join([f"{mem.memory_key}: {mem.memory_value}" for mem in user.memories])
    avatar_memory_str = ", ".join([f"{mem.memory_key}: {mem.memory_value}" for mem in avatar.memories])
    conversation_history_messages = [
        msg for turn in interaction_history 
        for role, content in [("user", turn.user_input_text), ("assistant", turn.agent_response_text)] 
        if content for msg in [{"role": role, "content": content}]
    ]

    # 2. Stage 1: Generate Raw Content
    try:
        content_provider = avatar.content_provider
        content_client = get_openai_client(content_provider.provider_endpoint, content_provider.api_key_env_var)
        
        system_prompt_parts = ["You are a helpful, factual AI assistant."]
        user_context_parts = []
        if interaction.perceived_user_affect:
            user_context_parts.append(f"{interaction.perceived_user_affect}")
        if user_memory_str:
             user_context_parts.append(f"You remember the following about them: {user_memory_str}.")

        if user_context_parts:
            system_prompt_parts.append(f"For context: {' '.join(user_context_parts)}.")

        messages_for_content = [
            {"role": "system", "content": " ".join(system_prompt_parts)},
            *conversation_history_messages,
            {"role": "user", "content": interaction.user_input_text}
        ]

        response = content_client.chat.completions.create(model=content_provider.model_name, messages=messages_for_content)
        raw_content_response = response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error from Content Provider: {e}")
        raise HTTPException(status_code=500, detail="Failed to get response from content provider.")

    # 3. Stage 2: Adapt Response to Persona
    try:
        expression_provider = avatar.expression_provider
        expression_client = get_openai_client(expression_provider.provider_endpoint, expression_provider.api_key_env_var)
        
        system_prompt = (
            "You are an expert at rephrasing text to match a specific persona for a Text-to-Speech engine. "
            "Your task is to rewrite the provided text according to the persona instructions. "
            "Crucially, the output must be a single, continuous block of verbatim text, exactly as someone would speak it. "
            "Do not use any markdown formatting, lists, bullet points, or any other non-verbal text formatting."
        )

        prompt_parts = [expression_provider.base_prompt_template]
        prompt_parts.append(f"Your Name is: {avatar.avatar_name}.")
        if avatar.description:
            prompt_parts.append(f"Your Description: {avatar.description}.")
        if avatar_memory_str:
            prompt_parts.append(f"Remember your own key traits: {avatar_memory_str}.")
        if expression_provider.reference_text:
            prompt_parts.append(f"\n\nHere is an example of your desired writing style:\n{expression_provider.reference_text}")
        prompt_parts.append(f"\n\nNow, rephrase the following text according to your persona.\nOriginal Text: \"{raw_content_response}\"\n\nRephrased Text:")
        
        expression_prompt = "".join(prompt_parts)
        
        messages_for_expression = [
            {"role": "system", "content": system_prompt},
            *conversation_history_messages,
            {"role": "user", "content": expression_prompt}
        ]

        response = expression_client.chat.completions.create(model=expression_provider.model_name, messages=messages_for_expression)
        final_response_text = response.choices[0].message.content.strip().replace("*", "") # Clean up common markdown artifacts
    except Exception as e:
        logger.error(f"Error from Expression Provider: {e}")
        raise HTTPException(status_code=500, detail="Failed to get response from expression provider.")
    
    try:
        interaction.raw_content_response = raw_content_response
        interaction.agent_response_text = final_response_text
        db.commit()
        logger.info(f"Successfully saved generated text to interaction {request.interaction_id}")
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to save text to database for interaction {request.interaction_id}: {e}")
        raise HTTPException(status_code=500, detail="Database update failed after text generation.")

    return ConversationalResponse(
        raw_content_response=raw_content_response,
        final_response_text=final_response_text
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
