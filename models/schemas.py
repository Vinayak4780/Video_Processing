from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class VideoUploadRequest(BaseModel):
    filename: str
    description: Optional[str] = None

class EventDetection(BaseModel):
    timestamp: float
    event_type: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    description: str

class VideoAnalysis(BaseModel):
    video_id: str
    filename: str
    duration: float
    total_frames: int
    fps: float
    events: List[EventDetection]
    summary: str
    violations: List[str]
    processed_at: datetime

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime
    
class ChatRequest(BaseModel):
    session_id: str
    message: str
    video_id: Optional[str] = None

class ChatResponse(BaseModel):
    session_id: str
    response: str
    context_used: bool
    timestamp: datetime
    processing_time_ms: Optional[int] = None  # Round 2 latency tracking

class ConversationSession(BaseModel):
    session_id: str
    video_id: Optional[str] = None
    messages: List[ChatMessage]
    created_at: datetime
    last_active: datetime
