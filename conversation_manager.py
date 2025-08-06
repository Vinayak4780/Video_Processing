from groq import Groq
from typing import List, Dict, Optional
from models.schemas import ChatMessage, ConversationSession, VideoAnalysis
from utils.helpers import Logger, generate_session_id
from datetime import datetime
import json
import os

class ConversationManager:
    def __init__(self, api_key: str, model: str = "llama3-8b-8192"):
        """Initialize Groq client for conversation management"""
        self.logger = Logger()
        
        try:
            # Initialize Groq client with the API key
            self.client = Groq(api_key=api_key)
            self.model = model
            self.logger.info("Groq client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Groq client: {e}")
            raise ValueError(f"Failed to initialize Groq client: {e}")
        
        # Store active conversations
        self.conversations: Dict[str, ConversationSession] = {}
        
        # Store video analyses for context
        self.video_analyses: Dict[str, VideoAnalysis] = {}
        
        # System prompt for the assistant
        self.system_prompt = """You are an intelligent video analysis assistant specializing in traffic scene understanding and safety monitoring. You have the ability to:

1. Analyze video content and detect objects, events, and potential violations
2. Provide detailed explanations about traffic scenarios
3. Answer questions about detected events, timestamps, and safety concerns
4. Engage in natural multi-turn conversations while maintaining context

When responding:
- Be conversational and helpful
- Provide specific details when available (timestamps, object types, locations)
- Explain traffic rules and safety implications when relevant
- Ask clarifying questions if needed
- Maintain context from previous messages in the conversation

If asked about video content, refer to the provided video analysis data to give accurate, specific answers."""

    def create_session(self, video_id: Optional[str] = None) -> str:
        """Create a new conversation session"""
        session_id = generate_session_id()
        
        session = ConversationSession(
            session_id=session_id,
            video_id=video_id,
            messages=[],
            created_at=datetime.now(),
            last_active=datetime.now()
        )
        
        self.conversations[session_id] = session
        self.logger.info(f"Created new conversation session: {session_id}")
        
        return session_id
    
    def add_video_analysis(self, video_analysis: VideoAnalysis):
        """Store video analysis for conversation context"""
        self.video_analyses[video_analysis.video_id] = video_analysis
        self.logger.info(f"Added video analysis for video: {video_analysis.video_id}")
    
    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Get conversation session by ID"""
        return self.conversations.get(session_id)
    
    def send_message(self, session_id: str, message: str) -> str:
        """Send message and get response from the assistant"""
        session = self.conversations.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        # Add user message to conversation
        user_message = ChatMessage(
            role="user",
            content=message,
            timestamp=datetime.now()
        )
        session.messages.append(user_message)
        
        # Prepare context for the assistant
        context = self._prepare_context(session)
        
        # Generate response using Groq
        response_content = self._generate_response(context, message, session)
        
        # Add assistant response to conversation
        assistant_message = ChatMessage(
            role="assistant",
            content=response_content,
            timestamp=datetime.now()
        )
        session.messages.append(assistant_message)
        
        # Update session activity
        session.last_active = datetime.now()
        
        self.logger.info(f"Processed message in session {session_id}")
        
        return response_content
    
    def _prepare_context(self, session: ConversationSession) -> Dict:
        """Prepare context for the conversation"""
        context = {
            "conversation_history": [],
            "video_analysis": None
        }
        
        # Add conversation history (last 10 messages to manage token limits)
        recent_messages = session.messages[-10:] if len(session.messages) > 10 else session.messages
        for msg in recent_messages:
            context["conversation_history"].append({
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat()
            })
        
        # Add video analysis if available
        if session.video_id and session.video_id in self.video_analyses:
            video_analysis = self.video_analyses[session.video_id]
            context["video_analysis"] = {
                "filename": video_analysis.filename,
                "duration": video_analysis.duration,
                "summary": video_analysis.summary,
                "events_count": len(video_analysis.events),
                "violations_count": len(video_analysis.violations),
                "key_events": [
                    {
                        "timestamp": event.timestamp,
                        "type": event.event_type,
                        "description": event.description,
                        "confidence": event.confidence
                    } for event in video_analysis.events[:5]  # Top 5 events
                ],
                "violations": video_analysis.violations[:5]  # Top 5 violations
            }
        
        return context
    
    def _generate_response(self, context: Dict, message: str, session: ConversationSession) -> str:
        """Generate response using Groq API"""
        try:
            # Prepare messages for Groq API
            messages = [
                {"role": "system", "content": self.system_prompt}
            ]
            
            # Add video context if available
            if context["video_analysis"]:
                video_context = f"""
Video Analysis Context:
- File: {context['video_analysis']['filename']}
- Duration: {context['video_analysis']['duration']:.2f} seconds
- Events detected: {context['video_analysis']['events_count']}
- Violations found: {context['video_analysis']['violations_count']}

Summary:
{context['video_analysis']['summary']}

Recent Events:
{chr(10).join([f"- {event['timestamp']:.1f}s: {event['description']}" for event in context['video_analysis']['key_events']])}

Violations:
{chr(10).join([f"- {violation}" for violation in context['video_analysis']['violations']])}
"""
                messages.append({"role": "system", "content": video_context})
            
            # Add conversation history
            for msg in context["conversation_history"]:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            # Add current message
            messages.append({"role": "user", "content": message})
            
            # Generate response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error while processing your message. Please try again."
    
    def get_conversation_history(self, session_id: str) -> List[Dict]:
        """Get conversation history for a session"""
        session = self.conversations.get(session_id)
        if not session:
            return []
        
        return [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat()
            } for msg in session.messages
        ]
    
    def clear_session(self, session_id: str) -> bool:
        """Clear a conversation session"""
        if session_id in self.conversations:
            del self.conversations[session_id]
            self.logger.info(f"Cleared session: {session_id}")
            return True
        return False
    
    def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs"""
        return list(self.conversations.keys())
