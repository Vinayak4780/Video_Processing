import streamlit as st
import requests
import json
from datetime import datetime
import time
import os

# Page configuration
st.set_page_config(
    page_title="Video Analysis Chat Assistant",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
API_BASE_URL = "http://localhost:8000"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        color: white;
        text-align: center;
        margin: 0;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .user-message {
        background-color: #e3f2fd;
        margin-left: 2rem;
    }
    
    .assistant-message {
        background-color: #f5f5f5;
        margin-right: 2rem;
    }
    
    .video-summary {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ff9800;
    }
    
    .status-success {
        color: #4caf50;
        font-weight: bold;
    }
    
    .status-error {
        color: #f44336;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎥 Video Analysis Chat Assistant</h1>
        <p style="text-align: center; color: white; margin: 0;">
            Upload videos, analyze traffic scenes, and chat about the results
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'session_id' not in st.session_state:
        st.session_state.session_id = None
    if 'video_id' not in st.session_state:
        st.session_state.video_id = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'video_processed' not in st.session_state:
        st.session_state.video_processed = False
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # Video Upload Section
        st.subheader("📤 Upload Video")
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Maximum duration: 2 minutes"
        )
        
        if uploaded_file is not None and not st.session_state.video_processed:
            if st.button("🔍 Process Video", type="primary"):
                process_video(uploaded_file)
        
        # Session Management
        st.subheader("💬 Chat Session")
        if st.button("🆕 New Session"):
            create_new_session()
        
        if st.session_state.session_id:
            st.success(f"Session: {st.session_state.session_id[:8]}...")
            if st.button("🗑️ Clear Chat"):
                clear_chat()
        
        # Video Info
        if st.session_state.video_id:
            st.subheader("📹 Current Video")
            st.info(f"Video ID: {st.session_state.video_id[:8]}...")
            
            if st.button("📊 Show Video Summary"):
                show_video_summary()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Chat Interface")
        
        # Display chat messages
        display_chat_messages()
    
    with col2:
        st.header("📋 Quick Actions")
        
        # Sample questions
        if st.session_state.video_processed:
            st.subheader("💡 Sample Questions")
            sample_questions = [
                "What vehicles were detected in the video?",
                "Were there any traffic violations?",
                "Can you summarize the key events?",
                "What safety recommendations do you have?",
                "How many pedestrians were detected?",
                "What was the most concerning event?"
            ]
            
            for question in sample_questions:
                if st.button(f"❓ {question}", key=question):
                    if st.session_state.session_id:
                        send_message(question)
        
        # System status
        st.subheader("🔧 System Status")
        check_api_status()
    
    # Chat input (outside columns to avoid Streamlit error)
    if st.session_state.session_id:
        user_input = st.chat_input("Ask me anything about the video...")
        if user_input:
            send_message(user_input)
    else:
        st.info("👈 Create a new session to start chatting!")

def process_video(uploaded_file):
    """Process uploaded video"""
    with st.spinner("🔄 Processing video... This may take a few minutes."):
        try:
            # Prepare file for upload
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            
            # Upload and process video
            response = requests.post(f"{API_BASE_URL}/upload-video", files=files)
            
            if response.status_code == 200:
                result = response.json()
                st.session_state.video_id = result['video_id']
                st.session_state.video_processed = True
                
                # Create new session with video context
                create_new_session(st.session_state.video_id)
                
                # Display success message
                st.success("✅ Video processed successfully!")
                
                # Display summary
                st.markdown(f"""
                <div class="video-summary">
                    <h4>📊 Processing Results</h4>
                    <ul>
                        <li><strong>Duration:</strong> {result['duration']:.1f} seconds</li>
                        <li><strong>Events detected:</strong> {result['events_detected']}</li>
                        <li><strong>Violations found:</strong> {result['violations_found']}</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                # Auto-send initial message
                initial_message = "Please provide a summary of the video analysis."
                send_message(initial_message)
                
            else:
                st.error(f"❌ Error processing video: {response.text}")
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

def create_new_session(video_id=None):
    """Create a new chat session"""
    try:
        data = {}
        if video_id:
            data['video_id'] = video_id
        
        response = requests.post(f"{API_BASE_URL}/create-session", json=data)
        
        if response.status_code == 200:
            result = response.json()
            st.session_state.session_id = result['session_id']
            st.session_state.messages = []
            st.success("✅ New session created!")
        else:
            st.error(f"❌ Error creating session: {response.text}")
            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

def send_message(message):
    """Send message to chat assistant"""
    if not st.session_state.session_id:
        st.error("❌ No active session. Please create a new session first.")
        return
    
    # Add user message to display
    st.session_state.messages.append({
        'role': 'user',
        'content': message,
        'timestamp': datetime.now().isoformat()
    })
    
    try:
        # Send to API
        data = {
            'session_id': st.session_state.session_id,
            'message': message,
            'video_id': st.session_state.video_id
        }
        
        with st.spinner("🤔 Thinking..."):
            response = requests.post(f"{API_BASE_URL}/chat", json=data)
        
        if response.status_code == 200:
            result = response.json()
            
            # Add assistant response to display
            st.session_state.messages.append({
                'role': 'assistant',
                'content': result['response'],
                'timestamp': result['timestamp']
            })
            
            # Refresh the page to show new messages
            st.rerun()
            
        else:
            st.error(f"❌ Error sending message: {response.text}")
            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

def display_chat_messages():
    """Display chat messages"""
    if not st.session_state.messages:
        st.info("👋 Start a conversation by typing a message below!")
        return
    
    for message in st.session_state.messages:
        if message['role'] == 'user':
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>🧑 You:</strong><br>
                {message['content']}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message assistant-message">
                <strong>🤖 Assistant:</strong><br>
                {message['content']}
            </div>
            """, unsafe_allow_html=True)

def clear_chat():
    """Clear chat session"""
    if st.session_state.session_id:
        try:
            response = requests.delete(f"{API_BASE_URL}/conversation/{st.session_state.session_id}")
            if response.status_code == 200:
                st.session_state.messages = []
                st.session_state.session_id = None
                st.success("✅ Chat cleared!")
                st.rerun()
            else:
                st.error(f"❌ Error clearing chat: {response.text}")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

def show_video_summary():
    """Show detailed video summary"""
    if not st.session_state.video_id:
        st.error("❌ No video processed yet.")
        return
    
    try:
        response = requests.get(f"{API_BASE_URL}/video-summary/{st.session_state.video_id}")
        
        if response.status_code == 200:
            summary = response.json()
            
            st.markdown(f"""
            ### 📊 Video Analysis Summary
            
            **File:** {summary['filename']}  
            **Duration:** {summary['duration']:.1f} seconds  
            **Events Detected:** {summary['events_count']}  
            **Violations Found:** {summary['violations_count']}  
            **Processed:** {summary['processed_at']}
            
            ---
            
            {summary['summary']}
            """)
        else:
            st.error(f"❌ Error fetching summary: {response.text}")
            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

def check_api_status():
    """Check API health status"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        
        if response.status_code == 200:
            health = response.json()
            st.markdown(f"""
            <div class="status-success">
                ✅ API Status: Healthy<br>
                📊 Active Sessions: {health['active_sessions']}<br>
                🎥 Processed Videos: {health['processed_videos']}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-error">❌ API Status: Error</div>', unsafe_allow_html=True)
            
    except Exception as e:
        st.markdown('<div class="status-error">❌ API Status: Offline</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
