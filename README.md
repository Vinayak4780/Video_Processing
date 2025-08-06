# Video Analysis Chat Assistant

A sophisticated agentic chat assistant for visual understanding that processes video input, recognizes events, summarizes content, and engages in multi-turn conversations.

## Features

- **Video Event Recognition**: Uses YOLO for real-time object detection and event recognition in video streams
- **Content Summarization**: Generates comprehensive summaries highlighting key events and guideline adherence/violations
- **Multi-Turn Conversations**: Supports natural conversations with context retention using Groq API
- **Traffic Scene Analysis**: Specialized for traffic monitoring with vehicle tracking, pedestrian detection, and violation detection

## Tech Stack

- **Backend**: FastAPI for high-performance API endpoints
- **Computer Vision**: YOLOv8 (Ultralytics) for object detection and tracking
- **AI/LLM**: Groq API for conversational AI and text generation
- **Video Processing**: OpenCV for video stream processing
- **Frontend**: Streamlit for interactive web interface

## Installation

1. Clone the repository and navigate to the project directory
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Create .env file with your Groq API key
GROQ_API_KEY=your_groq_api_key_here
```

## Usage

### Start the Backend API
```bash
python main.py
```

### Start the Streamlit Frontend
```bash
streamlit run frontend.py
```

## Project Structure

- `main.py` - FastAPI backend server
- `video_processor.py` - Video processing and YOLO integration
- `conversation_manager.py` - Multi-turn conversation handling
- `event_detector.py` - Event recognition and analysis
- `frontend.py` - Streamlit web interface
- `models/` - Data models and schemas
- `utils/` - Utility functions

## API Endpoints

- `POST /upload-video` - Upload and process video file
- `POST /chat` - Send message to chat assistant
- `GET /video-summary/{video_id}` - Get video analysis summary
- `GET /conversation/{session_id}` - Get conversation history
