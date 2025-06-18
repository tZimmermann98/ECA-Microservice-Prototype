import logging
from sqlalchemy.orm import Session
from utils.db import (
    engine, Base, User, AudioProvider, VideoProvider, 
    ExpressionProvider, ContentProvider, PerceptionProvider, Avatar,
    UserMemory, AvatarMemory, Session as DbSession, Interaction
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_example_data(db: Session):
    """Populate the database with initial example data."""
    logger.info("Seeding database with initial data...")

    # Check if a user already exists to prevent duplicate entries
    if db.query(User).first():
        logger.info("Database already seeded. Skipping.")
        return

    # 1. Create Providers
    logger.info("Creating providers...")
    content_prov = ContentProvider(
        provider_name="OpenAI_GPT4o",
        provider_endpoint="https://api.openai.com/v1",
        model_name="gpt-4o",
        api_key_env_var="OPENAI_API_KEY"
    )
    expr_prov = ExpressionProvider(
        provider_name="PersonaAdapter",
        provider_endpoint="https://api.openai.com/v1",
        model_name="gpt-4o",
        api_key_env_var="OPENAI_API_KEY",
        base_prompt_template="You are an expert in mimicking other expression styles. Rephrase the provided text to be professional and concise.",
        reference_text="Design science research is a problem-solving paradigm. It begins by identifying relevant, real-world problems. The next step involves a rigorous review of the existing knowledge base to understand what is already known. Often, existing solutions only partially address the problem, requiring researchers to design novel artifacts—constructs, models, methods, or instantiations—that are fit for the new context. This design process is not routine; it requires research to create innovative solutions. The proposed design is then evaluated to determine its utility and efficacy, which leads to a very iterative process of refinement until a satisfactory solution is achieved."
    )
    audio_prov = AudioProvider(
        provider_name="ExampleTTSProvider",
        provider_endpoint="https://api.fish.audio/v1/tts",
        # Replace this with a real voice_id from your TTS provider account
        provider_voice_id="<your_tts_provider_voice_id>",
        api_key_env_var="FISHAUDIO_API_KEY",
    )
    video_prov = VideoProvider(
        provider_name="ExampleVideoProvider",
        provider_endpoint="https://api.heygen.com/v2/video/generate",
        # Replace this with a real talking_photo_id from your video provider account
        provider_avatar_id="<your_video_provider_avatar_id>",
        api_key_env_var="HEYGEN_API_KEY"
    )
    perc_prov = PerceptionProvider(
        provider_name="LocalPerception",
        provider_endpoint="http://perception-service:8000/v1/analyze"
    )
    db.add_all([content_prov, expr_prov, audio_prov, video_prov, perc_prov])
    db.commit()

    # 2. Create an Avatar
    logger.info("Creating default avatar...")
    assistant_avatar = Avatar(
        avatar_name="Default Professor",
        description="This is the avatar of a digital professor of Information Systems. He is designed to assist with academic inquiries and explain complex concepts.",
        content_provider_id=content_prov.provider_id,
        expression_provider_id=expr_prov.provider_id,
        audio_provider_id=audio_prov.provider_id,
        video_provider_id=video_prov.provider_id,
        perception_provider_id=perc_prov.provider_id
    )
    db.add(assistant_avatar)
    db.commit()
    
    # 3. Create an Admin User
    logger.info("Creating admin user...")
    admin_user = User(
        user_email="admin@example.com",
        user_name="Admin",
        role="admin"
    )
    db.add(admin_user)
    db.commit()

    # 4. Create Memory for the Avatar
    logger.info("Creating avatar memory...")
    avatar_mem1 = AvatarMemory(avatar_id=assistant_avatar.avatar_id, memory_key="purpose", memory_value="To assist users with their research.")
    avatar_mem2 = AvatarMemory(avatar_id=assistant_avatar.avatar_id, memory_key="personality", memory_value="professional, helpful, concise")
    avatar_mem3 = AvatarMemory(avatar_id=assistant_avatar.avatar_id, memory_key="avatar_name", memory_value="Professor")
    db.add_all([avatar_mem1, avatar_mem2, avatar_mem3])

    # 5. Create Memory for the User
    logger.info("Creating user memory...")
    user_mem1 = UserMemory(user_id=admin_user.user_id, memory_key="user_name", memory_value="Default User")
    user_mem2 = UserMemory(user_id=admin_user.user_id, memory_key="last_topic", memory_value="Design Science Research")
    db.add_all([user_mem1, user_mem2])
    
    # 6. Create a Session
    logger.info("Creating a session...")
    new_session = DbSession(
        user_id=admin_user.user_id,
        avatar_id=assistant_avatar.avatar_id
    )
    db.add(new_session)
    db.commit()

    # 7. Create a historical Interaction in that Session
    logger.info("Creating a historical interaction...")
    historical_interaction = Interaction(
        session_id=new_session.session_id,
        user_input_text="Hello, what was the last topic we discussed?",
        perceived_user_affect="Visually, the user appears Neutral.",
        raw_content_response="Based on your user memory, the last topic you discussed was Design Science Research.",
        agent_response_text="Hello. The last topic we discussed was Design Science Research.",
        generated_audio_url="interaction_1.mp3",
        video_provider_task_id="<example_task_id>",
        status="completed",
        generated_video_url="<example_video_s3_key>"
    )
    db.add(historical_interaction)
    
    db.commit()
    logger.info("Database seeding completed successfully!")

if __name__ == "__main__":
    logger.info("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    
    db_session = Session(engine)
    try:
        create_example_data(db_session)
    finally:
        db_session.close()
