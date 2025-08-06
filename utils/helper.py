import os
import uuid
from datetime import datetime
from typing import Dict, Any

def generate_session_id() -> str:
    """Generate a unique session ID"""
    return str(uuid.uuid4())

def generate_video_id() -> str:
    """Generate a unique video ID"""
    return str(uuid.uuid4())

def ensure_upload_dir(upload_dir: str) -> None:
    """Ensure upload directory exists"""
    os.makedirs(upload_dir, exist_ok=True)

def format_timestamp(seconds: float) -> str:
    """Format timestamp in MM:SS format"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def calculate_video_metrics(total_frames: int, fps: float) -> Dict[str, Any]:
    """Calculate video metrics"""
    duration = total_frames / fps if fps > 0 else 0
    return {
        "duration": duration,
        "total_frames": total_frames,
        "fps": fps
    }

def validate_video_duration(duration: float, max_duration: int = 120) -> bool:
    """Validate video duration is within limits"""
    return duration <= max_duration

class Logger:
    """Simple logger utility"""
    
    @staticmethod
    def info(message: str):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] INFO: {message}")
    
    @staticmethod
    def error(message: str):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERROR: {message}")
    
    @staticmethod
    def warning(message: str):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] WARNING: {message}")
