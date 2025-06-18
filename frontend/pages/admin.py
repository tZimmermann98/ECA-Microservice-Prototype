import streamlit as st

from utils.db import (
    SessionLocal, User, Provider, AudioProvider, VideoProvider, 
    ExpressionProvider, ContentProvider, PerceptionProvider, Avatar
)

st.set_page_config(page_title="Admin Dashboard", layout="wide")
st.title("Admin Dashboard")

# 1. Get user email and database session
if "X-Forwarded-Email" not in st.context.headers:
    st.error("Authentication Error: User email not found. Please log in.")
    st.stop()

email = st.context.headers["X-Forwarded-Email"]
db = SessionLocal()

# 2. Check for Admin Role
user = db.query(User).filter(User.user_email == email).first()
if not user or user.role != 'admin':
    st.error("🚫 Access Denied: You do not have permission to view this page.")
    st.stop()

st.success(f"Welcome, Admin {user.user_name}!")

st.header("Manage Service Providers")

# Expander for Creating a new provider
with st.expander("Create New Provider"):
    with st.form("provider_form", clear_on_submit=True):
        st.subheader("Add a New Provider")
        provider_type = st.selectbox("Provider Type", ["ContentProvider", "ExpressionProvider", "AudioProvider", "VideoProvider", "PerceptionProvider"])
        provider_name = st.text_input("Provider Name (e.g., 'HeyGenTalkingPhoto')")
        provider_endpoint = st.text_input("Provider Endpoint (API URL)")
        api_key_env_var = st.text_input("API Key Environment Variable (e.g., 'HEYGEN_API_KEY')")
        
        # Additional fields based on provider type
        model_name = st.text_input("Model Name (e.g., 'gpt-4o')", key="new_provider_model_name") if provider_type in ["ContentProvider", "ExpressionProvider"] else None
        provider_voice_id = st.text_input("Provider Voice ID", key="new_provider_voice_id") if provider_type == "AudioProvider" else None
        provider_avatar_id = st.text_input("Provider Avatar/Photo ID", key="new_provider_avatar_id") if provider_type == "VideoProvider" else None
        base_prompt_template = st.text_area("Base Prompt Template", key="new_prompt") if provider_type == "ExpressionProvider" else None
        reference_text = st.text_area("Reference Text", key="new_ref_text") if provider_type == "ExpressionProvider" else None

        if st.form_submit_button("Create Provider"):
            try:
                new_provider = None
                if provider_type == "ContentProvider":
                    new_provider = ContentProvider(provider_name=provider_name, provider_endpoint=provider_endpoint, api_key_env_var=api_key_env_var, model_name=model_name)
                elif provider_type == "ExpressionProvider":
                    new_provider = ExpressionProvider(provider_name=provider_name, provider_endpoint=provider_endpoint, api_key_env_var=api_key_env_var, model_name=model_name, base_prompt_template=base_prompt_template, reference_text=reference_text)
                elif provider_type == "AudioProvider":
                    new_provider = AudioProvider(provider_name=provider_name, provider_endpoint=provider_endpoint, api_key_env_var=api_key_env_var, provider_voice_id=provider_voice_id)
                elif provider_type == "VideoProvider":
                    new_provider = VideoProvider(provider_name=provider_name, provider_endpoint=provider_endpoint, api_key_env_var=api_key_env_var, provider_avatar_id=provider_avatar_id)
                elif provider_type == "PerceptionProvider":
                    new_provider = PerceptionProvider(provider_name=provider_name, provider_endpoint=provider_endpoint, api_key_env_var=api_key_env_var)

                if new_provider:
                    db.add(new_provider)
                    db.commit()
                    st.success(f"Successfully created {provider_type}: {provider_name}")
                    st.rerun()
                else:
                    st.error("Invalid provider type selected.")
            except Exception as e:
                st.error(f"Failed to create provider: {e}")
                db.rollback()

# Expander for Editing/Deleting an existing provider
with st.expander("Edit or Delete Existing Provider"):
    st.subheader("Select a Provider to Manage")
    all_providers = db.query(Provider).all()
    if not all_providers:
        st.warning("No providers found in the database.")
    else:
        provider_options = {f"{p.provider_name} (ID: {p.provider_id}, Type: {p.provider_type})": p for p in all_providers}
        selected_provider_key = st.selectbox("Select Provider", options=provider_options.keys())
        selected_provider = provider_options[selected_provider_key]

        with st.form("edit_provider_form"):
            st.write(f"**Editing Provider ID: {selected_provider.provider_id}**")
            
            # Display fields for editing
            name = st.text_input("Provider Name", value=selected_provider.provider_name)
            endpoint = st.text_input("Provider Endpoint", value=selected_provider.provider_endpoint)
            api_key_var = st.text_input("API Key Env Var", value=selected_provider.api_key_env_var)
            
            model_name = st.text_input("Model Name", value=getattr(selected_provider, 'model_name', '')) if isinstance(selected_provider, (ContentProvider, ExpressionProvider)) else None
            voice_id = st.text_input("Provider Voice ID", value=getattr(selected_provider, 'provider_voice_id', '')) if isinstance(selected_provider, AudioProvider) else None
            avatar_id = st.text_input("Provider Avatar/Photo ID", value=getattr(selected_provider, 'provider_avatar_id', '')) if isinstance(selected_provider, VideoProvider) else None
            prompt_template = st.text_area("Base Prompt Template", value=getattr(selected_provider, 'base_prompt_template', '')) if isinstance(selected_provider, ExpressionProvider) else None
            ref_text = st.text_area("Reference Text", value=getattr(selected_provider, 'reference_text', '')) if isinstance(selected_provider, ExpressionProvider) else None

            col1, col2 = st.columns([1, 5])
            with col1:
                if st.form_submit_button("Save Changes", use_container_width=True):
                    try:
                        selected_provider.provider_name = name
                        selected_provider.provider_endpoint = endpoint
                        selected_provider.api_key_env_var = api_key_var
                        if model_name is not None: selected_provider.model_name = model_name
                        if voice_id is not None: selected_provider.provider_voice_id = voice_id
                        if avatar_id is not None: selected_provider.provider_avatar_id = avatar_id
                        if prompt_template is not None: selected_provider.base_prompt_template = prompt_template
                        if ref_text is not None: selected_provider.reference_text = ref_text
                        db.commit()
                        st.success("Provider updated successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to update provider: {e}")
                        db.rollback()

            with col2:
                if st.form_submit_button("Delete Provider", type="primary", use_container_width=True):
                    try:
                        db.delete(selected_provider)
                        db.commit()
                        st.success("Provider deleted successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to delete provider: {e}. It might be in use by an avatar.")
                        db.rollback()

st.header("Manage Avatars")

with st.expander("Create New Avatar"):
    with st.form("avatar_form", clear_on_submit=True):
        st.subheader("Add a New Avatar")
        avatar_name = st.text_input("Avatar Name")
        description = st.text_area("Avatar Description")
        
        # Fetch providers for dropdowns
        providers = db.query(Provider).all()
        content_providers = {f"{p.provider_name} (ID: {p.provider_id})": p.provider_id for p in providers if isinstance(p, ContentProvider)}
        expression_providers = {f"{p.provider_name} (ID: {p.provider_id})": p.provider_id for p in providers if isinstance(p, ExpressionProvider)}
        audio_providers = {f"{p.provider_name} (ID: {p.provider_id})": p.provider_id for p in providers if isinstance(p, AudioProvider)}
        video_providers = {f"{p.provider_name} (ID: {p.provider_id})": p.provider_id for p in providers if isinstance(p, VideoProvider)}
        perception_providers = {f"{p.provider_name} (ID: {p.provider_id})": p.provider_id for p in providers if isinstance(p, PerceptionProvider)}

        # Dropdowns to link to existing providers
        sel_content_key = st.selectbox("Content Provider", options=content_providers.keys())
        sel_expr_key = st.selectbox("Expression Provider", options=expression_providers.keys())
        sel_audio_key = st.selectbox("Audio Provider", options=audio_providers.keys())
        sel_video_key = st.selectbox("Video Provider", options=video_providers.keys())
        sel_perc_key = st.selectbox("Perception Provider", options=perception_providers.keys())

        if st.form_submit_button("Create Avatar"):
            try:
                new_avatar = Avatar(
                    avatar_name=avatar_name,
                    description=description,
                    content_provider_id=content_providers.get(sel_content_key),
                    expression_provider_id=expression_providers.get(sel_expr_key),
                    audio_provider_id=audio_providers.get(sel_audio_key),
                    video_provider_id=video_providers.get(sel_video_key),
                    perception_provider_id=perception_providers.get(sel_perc_key)
                )
                db.add(new_avatar)
                db.commit()
                st.success(f"Successfully created avatar: {avatar_name}")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to create avatar: {e}")
                db.rollback()

# Expander for Editing/Deleting an existing avatar
with st.expander("Edit or Delete Existing Avatar"):
    st.subheader("Select an Avatar to Manage")
    all_avatars = db.query(Avatar).all()
    if not all_avatars:
        st.warning("No avatars found in the database.")
    else:
        avatar_options = {f"{a.avatar_name} (ID: {a.avatar_id})": a for a in all_avatars}
        selected_avatar_key = st.selectbox("Select Avatar", options=avatar_options.keys())
        selected_avatar = avatar_options[selected_avatar_key]

        with st.form("edit_avatar_form"):
            st.write(f"**Editing Avatar ID: {selected_avatar.avatar_id}**")
            
            # Display fields for editing
            avatar_name = st.text_input("Avatar Name", value=selected_avatar.avatar_name)
            description = st.text_area("Avatar Description", value=selected_avatar.description)
            
            # Fetch providers for dropdowns
            providers = db.query(Provider).all()
            content_providers = {f"{p.provider_name} (ID: {p.provider_id})": p.provider_id for p in providers if isinstance(p, ContentProvider)}
            expression_providers = {f"{p.provider_name} (ID: {p.provider_id})": p.provider_id for p in providers if isinstance(p, ExpressionProvider)}
            audio_providers = {f"{p.provider_name} (ID: {p.provider_id})": p.provider_id for p in providers if isinstance(p, AudioProvider)}
            video_providers = {f"{p.provider_name} (ID: {p.provider_id})": p.provider_id for p in providers if isinstance(p, VideoProvider)}
            perception_providers = {f"{p.provider_name} (ID: {p.provider_id})": p.provider_id for p in providers if isinstance(p, PerceptionProvider)}

            # Set default index for selectboxes based on current avatar's providers
            def get_index(options, current_id):
                ids = list(options.values())
                return ids.index(current_id) if current_id in ids else 0

            # Dropdowns to link to existing providers
            sel_content = st.selectbox("Content Provider", options=content_providers.keys(), index=get_index(content_providers, selected_avatar.content_provider_id))
            sel_expr = st.selectbox("Expression Provider", options=expression_providers.keys(), index=get_index(expression_providers, selected_avatar.expression_provider_id))
            sel_audio = st.selectbox("Audio Provider", options=audio_providers.keys(), index=get_index(audio_providers, selected_avatar.audio_provider_id))
            sel_video = st.selectbox("Video Provider", options=video_providers.keys(), index=get_index(video_providers, selected_avatar.video_provider_id))
            sel_perc = st.selectbox("Perception Provider", options=perception_providers.keys(), index=get_index(perception_providers, selected_avatar.perception_provider_id))

            col1, col2 = st.columns([1, 5])
            with col1:
                if st.form_submit_button("Save Changes", use_container_width=True):
                    try:
                        selected_avatar.avatar_name = avatar_name
                        selected_avatar.description = description
                        selected_avatar.content_provider_id = content_providers[sel_content]
                        selected_avatar.expression_provider_id = expression_providers[sel_expr]
                        selected_avatar.audio_provider_id = audio_providers[sel_audio]
                        selected_avatar.video_provider_id = video_providers[sel_video]
                        selected_avatar.perception_provider_id = perception_providers[sel_perc]
                        db.commit()
                        st.success("Avatar updated successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to update avatar: {e}")
                        db.rollback()
            with col2:
                if st.form_submit_button("Delete Avatar", type="primary", use_container_width=True):
                    try:
                        db.delete(selected_avatar)
                        db.commit()
                        st.success("Avatar deleted successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to delete avatar: {e}")
                        db.rollback()

db.close()
