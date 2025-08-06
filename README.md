# Video Analysis Chat Assistant

An intelligent agentic chat assistant for visual understanding that processes video input, recognizes events, summarizes content, and engages in multi-turn conversations about traffic scenes and safety monitoring.

## Project Overview

This project implements a comprehensive video analysis system that combines computer vision with conversational AI to create an interactive assistant for traffic monitoring and safety analysis. The system processes video streams up to 2 minutes in duration, detects objects and events using YOLO (You Only Look Once), identifies potential traffic violations, and provides intelligent responses through a chat interface powered by Groq's large language models.

### Key Capabilities
- **Real-time Video Processing**: Analyzes video streams with frame-by-frame object detection
- **Event Recognition**: Identifies vehicles, pedestrians, traffic lights, and other objects
- **Violation Detection**: Detects speeding, red light violations, jaywalking, and unsafe behaviors
- **Intelligent Summarization**: Generates comprehensive reports of video content
- **Conversational AI**: Engages in natural multi-turn conversations about analysis results
- **Context Retention**: Maintains conversation history and video context across interactions

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    VIDEO ANALYSIS CHAT ASSISTANT                │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │    Backend      │    │   AI Services   │
│   (Streamlit)   │◄──►│   (FastAPI)     │◄──►│   (Groq API)    │
│                 │    │                 │    │                 │
│ • Video Upload  │    │ • REST APIs     │    │ • Chat LLM      │
│ • Chat Interface│    │ • Session Mgmt  │    │ • Context Aware │
│ • Status Display│    │ • File Handling │    │ • Multi-turn    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ Video Processor │
                       │    (YOLO)       │
                       │                 │
                       │ • Object Detect │
                       │ • Event Extract │
                       │ • Violation Det │
                       └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ Event Detector  │
                       │                 │
                       │ • Pattern Anal  │
                       │ • Safety Score  │
                       │ • Recommendations│
                       └─────────────────┘

Data Flow:
1. User uploads video via Streamlit frontend
2. FastAPI backend processes video with YOLO
3. Events and violations are detected and analyzed
4. Results stored and linked to chat session
5. User engages in conversation about analysis
6. Groq API provides intelligent responses with video context
```

## Tech Stack Justification

### Backend Technologies

**FastAPI (REST API Framework)**
- **Justification**: High-performance async framework ideal for AI/ML workloads
- **Benefits**: Automatic API documentation, type validation, async support for video processing
- **Scalability**: Built-in support for background tasks and concurrent request handling

**YOLOv8 (Computer Vision)**
- **Justification**: State-of-the-art real-time object detection with excellent accuracy
- **Benefits**: Fast inference, pre-trained models, comprehensive object recognition
- **Suitability**: Optimized for traffic scene analysis with vehicle and pedestrian detection

**Groq API (Conversational AI)**
- **Justification**: High-performance LLM inference with excellent speed and quality
- **Benefits**: Fast response times, context-aware conversations, reliable API
- **Models**: Supports multiple models including Llama3 for diverse use cases

### Supporting Technologies

**OpenCV (Video Processing)**
- **Justification**: Industry-standard computer vision library
- **Benefits**: Robust video I/O, frame manipulation, extensive documentation

**Streamlit (Frontend)**
- **Justification**: Rapid prototyping for ML applications with minimal code
- **Benefits**: Built-in widgets for file upload, chat interface, real-time updates

**Pydantic (Data Validation)**
- **Justification**: Type-safe data models with automatic validation
- **Benefits**: Runtime type checking, JSON serialization, clear API contracts

## Setup and Installation Instructions

### Prerequisites
- Python 3.8 or higher
- Groq API key ([Get one here](https://console.groq.com/))
- 4GB+ RAM (recommended for video processing)
- Windows/Linux/macOS

### Installation Steps

#### Option 1: Automated Setup (Recommended)

**Windows:**
```cmd
# Run the automated setup script
setup.bat
```

**Linux/macOS:**
```bash
# Make script executable and run
chmod +x setup.sh
./setup.sh
```

#### Option 2: Manual Installation

1. **Create Virtual Environment**
```bash
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

2. **Install Dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3. **Configure Environment**
```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your Groq API key
GROQ_API_KEY=your_actual_groq_api_key_here
VIDEO_UPLOAD_DIR=./uploads
MAX_VIDEO_DURATION=120
YOLO_MODEL_PATH=yolov8n.pt
```

4. **Download YOLO Model**
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### Running the Application

1. **Start Backend Server**
```bash
python main.py
```
*Server will be available at: http://localhost:8000*

2. **Start Frontend Interface** (in a new terminal)
```bash
streamlit run frontend.py
```
*Web interface will open at: http://localhost:8501*

3. **Verify Installation**
```bash
python test_api.py
```
