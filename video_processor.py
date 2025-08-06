import cv2
import numpy as np
import torch
import os
from typing import List, Dict, Tuple
from models.schemas import EventDetection
from utils.helpers import Logger, format_timestamp

# Monkey patch torch.load to use weights_only=False for YOLO compatibility
original_torch_load = torch.load

def patched_torch_load(f, map_location=None, pickle_module=None, weights_only=None, **kwargs):
    # Force weights_only=False for YOLO model loading
    return original_torch_load(f, map_location=map_location, pickle_module=pickle_module, weights_only=False, **kwargs)

torch.load = patched_torch_load

# Now import YOLO after patching
from ultralytics import YOLO

class VideoProcessor:
    def __init__(self, model_path: str = "yolov8n.pt"):
        """Initialize YOLO model for video processing"""
        self.logger = Logger()
        
        try:
            # Load YOLO model (will auto-download if not present)
            self.model = YOLO(model_path)
            self.logger.info(f"Successfully loaded YOLO model: {model_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            raise RuntimeError(f"Could not initialize YOLO model: {e}")
        
        # Define class names for traffic scene analysis
        self.traffic_classes = {
            'person': 0, 'bicycle': 1, 'car': 2, 'motorcycle': 3, 'airplane': 4,
            'bus': 5, 'train': 6, 'truck': 7, 'boat': 8, 'traffic light': 9,
            'fire hydrant': 10, 'stop sign': 11, 'parking meter': 12, 'bench': 13
        }
        
        # Define violation rules
        self.violation_rules = {
            'speed_threshold': 50,  # pixels per frame (approximate)
            'red_light_violation': True,
            'pedestrian_safety': True
        }
    
    def process_video(self, video_path: str) -> Dict:
        """Process video and extract events"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError("Could not open video file")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        self.logger.info(f"Processing video: {video_path}")
        self.logger.info(f"FPS: {fps}, Total frames: {total_frames}, Duration: {duration:.2f}s")
        
        events = []
        violations = []
        frame_count = 0
        
        # Track objects across frames for violation detection
        tracked_objects = {}
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_count / fps
            
            # Run YOLO detection
            results = self.model(frame)
            
            # Process detections
            frame_events, frame_violations = self._process_frame_detections(
                results, timestamp, frame_count, tracked_objects
            )
            
            events.extend(frame_events)
            violations.extend(frame_violations)
            
            frame_count += 1
            
            # Log progress every 30 frames
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                self.logger.info(f"Processing progress: {progress:.1f}%")
        
        cap.release()
        
        return {
            'duration': duration,
            'total_frames': total_frames,
            'fps': fps,
            'events': events,
            'violations': violations
        }
    
    def _process_frame_detections(self, results, timestamp: float, frame_count: int, 
                                tracked_objects: Dict) -> Tuple[List[EventDetection], List[str]]:
        """Process YOLO detections for a single frame"""
        events = []
        violations = []
        
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
                
            for box in boxes:
                # Extract detection data
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                
                # Get class name
                class_name = self.model.names[class_id]
                
                # Create event detection
                event = EventDetection(
                    timestamp=timestamp,
                    event_type=f"{class_name}_detected",
                    confidence=confidence,
                    bbox=bbox,
                    description=f"{class_name} detected with {confidence:.2f} confidence at {format_timestamp(timestamp)}"
                )
                
                # Only add high-confidence detections
                if confidence > 0.5:
                    events.append(event)
                
                # Check for violations
                frame_violations = self._check_violations(
                    class_name, bbox, timestamp, confidence, tracked_objects, frame_count
                )
                violations.extend(frame_violations)
        
        return events, violations
    
    def _check_violations(self, class_name: str, bbox: List[float], timestamp: float, 
                         confidence: float, tracked_objects: Dict, frame_count: int) -> List[str]:
        """Check for traffic violations"""
        violations = []
        
        # Calculate object center
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        # Create object ID based on position (simple tracking)
        object_id = f"{class_name}_{int(center_x//50)}_{int(center_y//50)}"
        
        # Track object movement
        if object_id in tracked_objects:
            prev_center = tracked_objects[object_id]['center']
            prev_frame = tracked_objects[object_id]['frame']
            
            # Calculate speed (pixels per frame)
            distance = np.sqrt((center_x - prev_center[0])**2 + (center_y - prev_center[1])**2)
            frame_diff = frame_count - prev_frame
            speed = distance / frame_diff if frame_diff > 0 else 0
            
            # Check for speeding
            if speed > self.violation_rules['speed_threshold'] and class_name in ['car', 'truck', 'bus', 'motorcycle']:
                violations.append(f"Potential speeding violation: {class_name} moving at high speed at {format_timestamp(timestamp)}")
        
        # Update tracking
        tracked_objects[object_id] = {
            'center': (center_x, center_y),
            'frame': frame_count,
            'class': class_name,
            'timestamp': timestamp
        }
        
        # Check for pedestrian safety violations
        if class_name == 'person':
            # Check if pedestrian is in road area (simplified logic)
            if self._is_in_road_area(bbox):
                violations.append(f"Pedestrian in road area detected at {format_timestamp(timestamp)}")
        
        return violations
    
    def _is_in_road_area(self, bbox: List[float]) -> bool:
        """Simple heuristic to determine if object is in road area"""
        # This is a simplified implementation
        # In a real system, you would use road segmentation or predefined areas
        center_y = (bbox[1] + bbox[3]) / 2
        # Assume road is in the middle portion of the frame
        return 0.3 < (center_y / 720) < 0.8  # Assuming 720p video
    
    def generate_summary(self, events: List[EventDetection], violations: List[str], 
                        duration: float) -> str:
        """Generate video summary from events and violations"""
        # Count events by type
        event_counts = {}
        for event in events:
            event_type = event.event_type.replace('_detected', '')
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        # Build summary
        summary_parts = []
        summary_parts.append(f"Video Analysis Summary (Duration: {format_timestamp(duration)})")
        summary_parts.append("=" * 50)
        
        # Object detection summary
        if event_counts:
            summary_parts.append("\nDetected Objects:")
            for obj_type, count in sorted(event_counts.items()):
                summary_parts.append(f"- {obj_type.title()}: {count} detections")
        
        # Violations summary
        if violations:
            summary_parts.append(f"\nViolations Detected ({len(violations)}):")
            for i, violation in enumerate(violations[:10], 1):  # Limit to first 10
                summary_parts.append(f"{i}. {violation}")
            if len(violations) > 10:
                summary_parts.append(f"... and {len(violations) - 10} more violations")
        else:
            summary_parts.append("\nNo violations detected.")
        
        # Key events timeline
        if events:
            summary_parts.append("\nKey Events Timeline:")
            # Get significant events (high confidence)
            significant_events = [e for e in events if e.confidence > 0.8][:5]
            for event in significant_events:
                summary_parts.append(f"- {format_timestamp(event.timestamp)}: {event.description}")
        
        return "\n".join(summary_parts)
