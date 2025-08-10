
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import shutil
from datetime import datetime
from typing import Optional
import uvicorn
import time
import json
import asyncio

# Import optimized modules
from models.schemas import (
    VideoUploadRequest, VideoAnalysis, ChatRequest, ChatResponse,
    EventDetection, ConversationSession
)
from video_processor import OptimizedVideoProcessor
from conversation_manager import ConversationManager
from event_detector import EventDetector
from utils.helpers import (
    Logger, generate_video_id, ensure_upload_dir, 
    validate_video_duration, calculate_video_metrics
)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Optional advanced features
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Initialize FastAPI app
app = FastAPI(
    title="Video Analysis Chat Assistant (Round 2)",
    description="High-performance agentic chat assistant for video analysis and traffic monitoring",
    version="2.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components with GPU optimization
logger = Logger()
video_processor = OptimizedVideoProcessor(
    enable_gpu=os.getenv("ENABLE_GPU", "true").lower() == "true",
    enable_tensorrt=os.getenv("ENABLE_TENSORRT", "true").lower() == "true",
    batch_size=int(os.getenv("BATCH_SIZE", "32"))
)
event_detector = EventDetector()

# Initialize conversation manager with Groq API
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    logger.error("GROQ_API_KEY not found in environment variables")
    raise ValueError("GROQ_API_KEY is required")

conversation_manager = ConversationManager(groq_api_key)

# Configuration
UPLOAD_DIR = os.getenv("VIDEO_UPLOAD_DIR", "./uploads")
MAX_VIDEO_DURATION = int(os.getenv("MAX_VIDEO_DURATION", "7200"))  # 120 min
MAX_FRAME_RATE = int(os.getenv("MAX_FRAME_RATE", "90"))
ENABLE_GPU = os.getenv("ENABLE_GPU_ACCELERATION", "true") == "true"

# Ensure upload directory exists
ensure_upload_dir(UPLOAD_DIR)

# Store video analyses in memory (in production, use a database)
video_analyses = {}

# Redis client
redis_client = None
if REDIS_AVAILABLE:
    try:
        redis_client = redis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379"),
            decode_responses=True
        )
        redis_client.ping()
        logger.info("Redis connection initialized")
    except Exception as e:
        logger.warning(f"Redis initialization failed: {e}")
        redis_client = None
        REDIS_AVAILABLE = False

# Prometheus metrics (disabled for development to avoid reload conflicts)
VIDEO_PROCESSING_TIME = None
CHAT_LATENCY = None

# Uncomment below for production with prometheus monitoring:
# if PROMETHEUS_AVAILABLE:
#     try:
#         VIDEO_PROCESSING_TIME = Histogram('video_processing_time_seconds', 'Time spent processing videos')
#         CHAT_LATENCY = Histogram('chat_latency_seconds', 'Chat response latency')
#     except Exception as e:
#         logger.warning(f"Prometheus metrics initialization failed: {e}")
#         VIDEO_PROCESSING_TIME = None
#         CHAT_LATENCY = None

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Video Analysis Chat Assistant API",
        "version": "1.0.0",
        "status": "running"
    }


# --- Round 2 Optimized Video Upload Endpoint ---
@app.post("/upload-video-optimized", response_model=dict)
async def upload_video_optimized(file: UploadFile = File(...)):
    """Upload and process video file with GPU, caching, and adaptive sampling"""
    start_time = time.time()
    try:
        # Validate file type
        if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
            raise HTTPException(status_code=400, detail="Invalid file type. Supported: mp4, avi, mov, mkv, webm")

        # Generate video ID and save file
        video_id = generate_video_id()
        file_path = os.path.join(UPLOAD_DIR, f"{video_id}_{file.filename}")

        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"Video uploaded: {file.filename} -> {file_path}")

        # Process video (GPU, adaptive sampling)
        logger.info("Starting optimized video processing...")
        processing_result = await video_processor.process_video_async(
            file_path,
            gpu=ENABLE_GPU,
            max_duration=MAX_VIDEO_DURATION,
            max_fps=MAX_FRAME_RATE
        )

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

        # Cache in Redis if available
        if REDIS_AVAILABLE and redis_client:
            try:
                redis_client.setex(
                    f"video_analysis:{video_id}",
                    3600,  # 1 hour TTL
                    json.dumps(video_analysis.__dict__, default=str)
                )
            except Exception as e:
                logger.warning(f"Redis cache failed: {e}")

        processing_time = time.time() - start_time
        if VIDEO_PROCESSING_TIME:
            VIDEO_PROCESSING_TIME.observe(processing_time)

        logger.info(f"Video processing completed for {video_id} in {processing_time:.2f}s")

        return {
            "video_id": video_id,
            "filename": file.filename,
            "duration": processing_result['duration'],
            "events_detected": len(processing_result['events']),
            "violations_found": len(processing_result['violations']),
            "summary": summary,
            "analysis": event_analysis,
            "processing_time": processing_time,
            "status": "completed"
        }

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        # Clean up file if it exists
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Video processing failed: {str(e)}")


# --- Round 2 Ultra-Fast Chat Endpoint ---
@app.post("/chat-fast", response_model=ChatResponse)
async def chat_fast(request: ChatRequest):
    """Send message to chat assistant with sub-1000ms latency"""
    start_time = time.time()
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
        response_content = await asyncio.to_thread(
            conversation_manager.send_message, session_id, request.message
        )

        latency = time.time() - start_time
        if CHAT_LATENCY:
            CHAT_LATENCY.observe(latency)

        return ChatResponse(
            session_id=session_id,
            response=response_content,
            context_used=request.video_id is not None,
            timestamp=datetime.now(),
            processing_time_ms=int(latency * 1000)
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


# --- Round 2 Cached Video Summary Endpoint ---
@app.get("/video-summary-cached/{video_id}")
async def get_video_summary_cached(video_id: str):
    """Get video analysis summary with Redis caching"""
    # Try Redis cache first
    if REDIS_AVAILABLE and redis_client:
        try:
            cached_analysis = redis_client.get(f"video_analysis:{video_id}")
            if cached_analysis:
                analysis_data = json.loads(cached_analysis)
                return {
                    "video_id": video_id,
                    "cached": True,
                    **analysis_data
                }
        except Exception as e:
            logger.warning(f"Redis cache retrieval failed: {e}")

    # Fallback to in-memory
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
        "processed_at": analysis.processed_at.isoformat(),
        "cached": False
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


# --- Round 2 Health Check with Metrics ---
@app.get("/health")
async def health_check():
    """Health check endpoint with metrics"""
    metrics = {}
    if PROMETHEUS_AVAILABLE:
        metrics['prometheus'] = generate_latest()
    if REDIS_AVAILABLE and redis_client:
        try:
            metrics['redis_ping'] = redis_client.ping()
        except Exception:
            metrics['redis_ping'] = False
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_sessions": len(conversation_manager.get_active_sessions()),
        "processed_videos": len(video_analyses),
        "metrics": metrics
    }


# --- Round 2 Performance Metrics Endpoint ---
@app.get("/performance-metrics")
async def get_performance_metrics():
    """Get detailed performance metrics"""
    try:
        metrics = video_processor.get_performance_metrics()
        return {
            "status": "success",
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


# --- Round 2 WebSocket Streaming Endpoint ---
@app.websocket("/ws/video-stream/{video_id}")
async def video_stream_ws(websocket: WebSocket, video_id: str):
    await websocket.accept()
    try:
        # Simulate streaming video events
        if video_id not in video_analyses:
            await websocket.send_json({"error": "Video not found"})
            await websocket.close()
            return
        analysis = video_analyses[video_id]
        for event in analysis.events:
            await websocket.send_json(event)
            await asyncio.sleep(0.1)  # Simulate real-time
        await websocket.send_json({"status": "completed"})
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for video {video_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await websocket.close()

if __name__ == "__main__":
    logger.info("Starting Video Analysis Chat Assistant API (Round 2)...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disabled to prevent Prometheus metrics collision
        log_level="info"
    )
