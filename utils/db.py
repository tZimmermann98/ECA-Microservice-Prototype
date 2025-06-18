import os
import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship, sessionmaker, declarative_base

Base = declarative_base()

DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "eca_prototype")

DB_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DB_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class Provider(Base):
    __tablename__ = 'provider'
    
    provider_id = Column(Integer, primary_key=True)
    provider_type = Column(String(50))
    provider_name = Column(String(255), nullable=False)
    provider_endpoint = Column(String(255))
    api_key_env_var = Column(String(255), nullable=True)
    model_name = Column(String(255), nullable=True)
    
    __mapper_args__ = {'polymorphic_identity': 'provider', 'polymorphic_on': provider_type}

class AudioProvider(Provider):
    __mapper_args__ = {'polymorphic_identity': 'audio_provider'}
    provider_voice_id = Column(String(255))
    audio_baseline_url = Column(String(255))

class VideoProvider(Provider):
    __mapper_args__ = {'polymorphic_identity': 'video_provider'}
    provider_avatar_id = Column(String(255))
    video_baseline_url = Column(String(255))

class ExpressionProvider(Provider):
    __mapper_args__ = {'polymorphic_identity': 'expression_provider'}
    base_prompt_template = Column(Text)
    reference_text = Column(Text)

class ContentProvider(Provider):
    __mapper_args__ = {'polymorphic_identity': 'content_provider'}

class PerceptionProvider(Provider):
    __mapper_args__ = {'polymorphic_identity': 'perception_provider'}

class Memory(Base):
    __tablename__ = 'memory'
    memory_id = Column(Integer, primary_key=True)
    memory_type = Column(String(50))
    memory_key = Column(String(255), nullable=False)
    memory_value = Column(Text)
    last_updated = Column(DateTime, default=datetime.datetime.utcnow)
    __mapper_args__ = {'polymorphic_identity': 'memory', 'polymorphic_on': memory_type}

class UserMemory(Memory):
    __mapper_args__ = {'polymorphic_identity': 'user_memory'}
    user_id = Column(Integer, ForeignKey('user.user_id'))
    user = relationship("User", back_populates="memories")

class AvatarMemory(Memory):
    __mapper_args__ = {'polymorphic_identity': 'avatar_memory'}
    avatar_id = Column(Integer, ForeignKey('avatar.avatar_id'))
    avatar = relationship("Avatar", back_populates="memories")

class User(Base):
    __tablename__ = 'user'
    user_id = Column(Integer, primary_key=True)
    user_email = Column(String(255), unique=True, nullable=False)
    user_name = Column(String(255))
    role = Column(String(50), default='user')
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    sessions = relationship("Session", back_populates="user")
    memories = relationship("UserMemory", back_populates="user", cascade="all, delete-orphan")

class Avatar(Base):
    __tablename__ = 'avatar'
    avatar_id = Column(Integer, primary_key=True)
    avatar_name = Column(String(255), nullable=False)
    description = Column(Text)
    audio_provider_id = Column(Integer, ForeignKey('provider.provider_id'))
    video_provider_id = Column(Integer, ForeignKey('provider.provider_id'))
    expression_provider_id = Column(Integer, ForeignKey('provider.provider_id'))
    content_provider_id = Column(Integer, ForeignKey('provider.provider_id'))
    perception_provider_id = Column(Integer, ForeignKey('provider.provider_id'))
    audio_provider = relationship("AudioProvider", foreign_keys=[audio_provider_id])
    video_provider = relationship("VideoProvider", foreign_keys=[video_provider_id])
    expression_provider = relationship("ExpressionProvider", foreign_keys=[expression_provider_id])
    content_provider = relationship("ContentProvider", foreign_keys=[content_provider_id])
    perception_provider = relationship("PerceptionProvider", foreign_keys=[perception_provider_id])
    sessions = relationship("Session", back_populates="avatar")
    memories = relationship("AvatarMemory", back_populates="avatar", cascade="all, delete-orphan")

class Session(Base):
    __tablename__ = 'session'
    session_id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('user.user_id'))
    avatar_id = Column(Integer, ForeignKey('avatar.avatar_id'))
    start_time = Column(DateTime, default=datetime.datetime.utcnow)
    end_time = Column(DateTime)
    user = relationship("User", back_populates="sessions")
    avatar = relationship("Avatar", back_populates="sessions")
    interactions = relationship("Interaction", back_populates="session", cascade="all, delete-orphan")

class Interaction(Base):
    __tablename__ = 'interaction'
    interaction_id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('session.session_id'))
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    video_provider_task_id = Column(String(255), nullable=True)
    status = Column(String(50), default='processing', nullable=False)
    user_input_audio_url = Column(String(255))
    user_input_video_url = Column(String(255))
    user_input_text = Column(Text)
    perceived_user_affect = Column(String(255))
    raw_content_response = Column(Text)
    agent_response_text = Column(Text)
    generated_audio_url = Column(String(255))
    generated_video_url = Column(String(255))
    session = relationship("Session", back_populates="interactions")
