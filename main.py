from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import shutil
from datetime import datetime
from typing import Optional
import uvicorn

# Import our modules
from models.schemas import (
    VideoUploadRequest, VideoAnalysis, ChatRequest, ChatResponse,
    EventDetection, ConversationSession
)
from video_processor import VideoProcessor
from conversation_manager import ConversationManager
from event_detector import EventDetector
from utils.helpers import (
    Logger, generate_video_id, ensure_upload_dir, 
    validate_video_duration, calculate_video_metrics
)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Video Analysis Chat Assistant",
    description="An agentic chat assistant for video analysis and traffic monitoring",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
logger = Logger()
video_processor = VideoProcessor()
event_detector = EventDetector()

# Initialize conversation manager with Groq API
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    logger.error("GROQ_API_KEY not found in environment variables")
    raise ValueError("GROQ_API_KEY is required")

conversation_manager = ConversationManager(groq_api_key)

# Configuration
UPLOAD_DIR = os.getenv("VIDEO_UPLOAD_DIR", "./uploads")
MAX_VIDEO_DURATION = int(os.getenv("MAX_VIDEO_DURATION", "120"))

# Ensure upload directory exists
ensure_upload_dir(UPLOAD_DIR)

# Store video analyses in memory (in production, use a database)
video_analyses = {}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Video Analysis Chat Assistant API",
        "version": "1.0.0",
        "status": "running"
    }

@app.post("/upload-video", response_model=dict)
async def upload_video(file: UploadFile = File(...)):
    """Upload and process video file"""
    try:
        # Validate file type
        if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            raise HTTPException(status_code=400, detail="Invalid file type. Supported: mp4, avi, mov, mkv")
        
        # Generate video ID and save file
        video_id = generate_video_id()
        file_path = os.path.join(UPLOAD_DIR, f"{video_id}_{file.filename}")
        
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Video uploaded: {file.filename} -> {file_path}")
        
        # Process video
        logger.info("Starting video processing...")
        processing_result = video_processor.process_video(file_path)
        
        # Validate video duration
        if not validate_video_duration(processing_result['duration'], MAX_VIDEO_DURATION):
            os.remove(file_path)  # Clean up
            raise HTTPException(
                status_code=400, 
                detail=f"Video duration exceeds maximum allowed duration of {MAX_VIDEO_DURATION} seconds"
            )
        
        # Analyze events
        logger.info("Analyzing events...")
        event_analysis = event_detector.analyze_events(
            processing_result['events'], 
            {
                'duration': processing_result['duration'],
                'total_frames': processing_result['total_frames'],
                'fps': processing_result['fps']
            }
        )
        
        # Generate summary
        summary = video_processor.generate_summary(
            processing_result['events'],
            processing_result['violations'],
            processing_result['duration']
        )
        
        # Create video analysis object
        video_analysis = VideoAnalysis(
            video_id=video_id,
            filename=file.filename,
            duration=processing_result['duration'],
            total_frames=processing_result['total_frames'],
            fps=processing_result['fps'],
            events=processing_result['events'],
            summary=summary,
            violations=processing_result['violations'],
            processed_at=datetime.now()
        )
        
        # Store analysis
        video_analyses[video_id] = video_analysis
        conversation_manager.add_video_analysis(video_analysis)
        
        logger.info(f"Video processing completed for {video_id}")
        
        return {
            "video_id": video_id,
            "filename": file.filename,
            "duration": processing_result['duration'],
            "events_detected": len(processing_result['events']),
            "violations_found": len(processing_result['violations']),
            "summary": summary,
            "analysis": event_analysis,
            "status": "completed"
        }
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        # Clean up file if it exists
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Video processing failed: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send message to chat assistant"""
    try:
        # Get or create session
        session = conversation_manager.get_session(request.session_id)
        if not session:
            # Create new session
            session_id = conversation_manager.create_session(request.video_id)
            session = conversation_manager.get_session(session_id)
        else:
            session_id = request.session_id
        
        # Generate response
        response_content = conversation_manager.send_message(session_id, request.message)
        
        return ChatResponse(
            session_id=session_id,
            response=response_content,
            context_used=request.video_id is not None,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

@app.post("/create-session")
async def create_session(video_id: Optional[str] = None):
    """Create a new chat session"""
    try:
        session_id = conversation_manager.create_session(video_id)
        return {
            "session_id": session_id,
            "video_id": video_id,
            "created_at": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error creating session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Session creation failed: {str(e)}")

@app.get("/video-summary/{video_id}")
async def get_video_summary(video_id: str):
    """Get video analysis summary"""
    if video_id not in video_analyses:
        raise HTTPException(status_code=404, detail="Video analysis not found")
    
    analysis = video_analyses[video_id]
    return {
        "video_id": video_id,
        "filename": analysis.filename,
        "duration": analysis.duration,
        "summary": analysis.summary,
        "events_count": len(analysis.events),
        "violations_count": len(analysis.violations),
        "processed_at": analysis.processed_at.isoformat()
    }

@app.get("/conversation/{session_id}")
async def get_conversation(session_id: str):
    """Get conversation history"""
    history = conversation_manager.get_conversation_history(session_id)
    if not history:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {
        "session_id": session_id,
        "messages": history,
        "message_count": len(history)
    }

@app.delete("/conversation/{session_id}")
async def clear_conversation(session_id: str):
    """Clear conversation session"""
    success = conversation_manager.clear_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {"message": "Session cleared successfully"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_sessions": len(conversation_manager.get_active_sessions()),
        "processed_videos": len(video_analyses)
    }

if __name__ == "__main__":
    logger.info("Starting Video Analysis Chat Assistant API...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
