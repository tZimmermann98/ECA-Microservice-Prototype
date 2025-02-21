# main.py
import os
import uuid
from argparse import Namespace

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse

import moviepy.editor as mp

# Import your inference functions.
# (Ensure that the PYTHONPATH is set appropriately so that 'scripts.inference' is importable.)
from scripts.inference import inference_process, merge_videos

app = FastAPI()

@app.post("/run_inference")
async def run_inference(
    image: UploadFile = File(...),
    audio: UploadFile = File(...),
    model_folder: str = Form(None),  # Optional; if not provided, the default in the config will be used.
):
    config = "default.yaml"
    # Create a unique working directory for this request.
    work_dir = os.path.join("./temp", str(uuid.uuid4()))
    os.makedirs(work_dir, exist_ok=True)
    
    # Save the uploaded image and audio files.
    image_path = os.path.join(work_dir, image.filename)
    audio_path = os.path.join(work_dir, audio.filename)
    
    with open(image_path, "wb") as f:
        f.write(await image.read())
    with open(audio_path, "wb") as f:
        f.write(await audio.read())
    
    # Construct a namespace object to mimic the command-line arguments expected by your inference script.
    # If model_folder is not provided, it will be omitted (thanks to filter_non_none in your code), so the default config value is used.
    args = Namespace(
        config=config,
        source_image=image_path,
        driving_audio=audio_path,
        pose_weight=1.0,
        face_weight=1.0,
        lip_weight=1.0,
        face_expand_ratio=1.2,
        audio_ckpt_dir=model_folder  # If model_folder is None, filter_non_none removes this key.
    )
    
    try:
        # Run your inference process. This should generate a folder (e.g., "./output_long/debug/<source_image_name>/")
        # containing the segmented video frames.
        save_seg_path = inference_process(args)
        # Merge the segmented frames into a single video.
        merge_videos(save_seg_path, os.path.join(os.path.dirname(save_seg_path), "merge_video.mp4"))
    except Exception as e:
        return {"error": f"Inference failed: {e}"}
    
    # Path to the merged video (without audio)
    merged_video_path = os.path.join(os.path.dirname(save_seg_path), "merge_video.mp4")
    
    try:
        # Load the video and the original audio using MoviePy.
        video_clip = mp.VideoFileClip(merged_video_path)
        audio_clip = mp.AudioFileClip(audio_path)
        
        # If the audio is longer than the video, trim it.
        if audio_clip.duration > video_clip.duration:
            audio_clip = audio_clip.subclip(0, video_clip.duration)
        
        # Combine the video with the audio.
        video_with_audio = video_clip.set_audio(audio_clip)
        
        # Write the final video to disk.
        final_video_path = os.path.join(work_dir, "final_video.mp4")
        video_with_audio.write_videofile(final_video_path, codec="libx264", audio_codec="aac")
        
        # Close the clips.
        video_clip.close()
        audio_clip.close()
        video_with_audio.close()
    except Exception as e:
        return {"error": f"Error adding audio: {e}"}
    
    # Return the final video file.
    return FileResponse(final_video_path, media_type="video/mp4", filename="final_video.mp4")
