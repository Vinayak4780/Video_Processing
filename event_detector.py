from typing import List, Dict, Any
from models.schemas import EventDetection, VideoAnalysis
import numpy as np
from utils.helpers import Logger, format_timestamp

class EventDetector:
    def __init__(self):
        """Initialize event detector with traffic rules and patterns"""
        self.logger = Logger()
        
        # Define traffic violation patterns
        self.violation_patterns = {
            'red_light_running': {
                'description': 'Vehicle passes through intersection during red light',
                'confidence_threshold': 0.7
            },
            'illegal_parking': {
                'description': 'Vehicle parked in prohibited area',
                'confidence_threshold': 0.6
            },
            'pedestrian_jaywalking': {
                'description': 'Pedestrian crosses outside designated area',
                'confidence_threshold': 0.5
            },
            'wrong_way_driving': {
                'description': 'Vehicle moving against traffic flow',
                'confidence_threshold': 0.8
            },
            'lane_violation': {
                'description': 'Vehicle crosses lane markings improperly',
                'confidence_threshold': 0.6
            }
        }
        
        # Traffic light states (simplified detection)
        self.traffic_light_states = ['red', 'yellow', 'green']
        
    def analyze_events(self, events: List[EventDetection], video_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze detected events to identify patterns and violations"""
        analysis = {
            'total_events': len(events),
            'event_distribution': self._calculate_event_distribution(events),
            'temporal_analysis': self._analyze_temporal_patterns(events, video_metrics),
            'violation_analysis': self._analyze_violations(events),
            'safety_score': self._calculate_safety_score(events),
            'recommendations': self._generate_recommendations(events)
        }
        
        return analysis
    
    def _calculate_event_distribution(self, events: List[EventDetection]) -> Dict[str, int]:
        """Calculate distribution of event types"""
        distribution = {}
        for event in events:
            event_type = event.event_type.replace('_detected', '')
            distribution[event_type] = distribution.get(event_type, 0) + 1
        
        return distribution
    
    def _analyze_temporal_patterns(self, events: List[EventDetection], video_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze temporal patterns in events"""
        if not events:
            return {'peak_activity': None, 'quiet_periods': [], 'activity_trend': 'stable'}
        
        # Divide video into time segments
        duration = video_metrics.get('duration', 0)
        num_segments = min(10, max(1, int(duration / 10)))  # 10-second segments
        segment_duration = duration / num_segments
        
        # Count events per segment
        segment_counts = [0] * num_segments
        for event in events:
            segment_idx = min(int(event.timestamp / segment_duration), num_segments - 1)
            segment_counts[segment_idx] += 1
        
        # Find peak activity period
        max_count = max(segment_counts)
        peak_segment = segment_counts.index(max_count)
        peak_start = peak_segment * segment_duration
        peak_end = (peak_segment + 1) * segment_duration
        
        # Find quiet periods (segments with low activity)
        avg_count = sum(segment_counts) / len(segment_counts)
        quiet_periods = []
        for i, count in enumerate(segment_counts):
            if count < avg_count * 0.5:  # Less than 50% of average
                start_time = i * segment_duration
                end_time = (i + 1) * segment_duration
                quiet_periods.append({
                    'start': format_timestamp(start_time),
                    'end': format_timestamp(end_time)
                })
        
        # Determine activity trend
        first_half = segment_counts[:len(segment_counts)//2]
        second_half = segment_counts[len(segment_counts)//2:]
        first_avg = sum(first_half) / len(first_half) if first_half else 0
        second_avg = sum(second_half) / len(second_half) if second_half else 0
        
        if second_avg > first_avg * 1.2:
            trend = 'increasing'
        elif second_avg < first_avg * 0.8:
            trend = 'decreasing'
        else:
            trend = 'stable'
        
        return {
            'peak_activity': {
                'start': format_timestamp(peak_start),
                'end': format_timestamp(peak_end),
                'event_count': max_count
            },
            'quiet_periods': quiet_periods,
            'activity_trend': trend,
            'events_per_segment': segment_counts
        }
    
    def _analyze_violations(self, events: List[EventDetection]) -> Dict[str, Any]:
        """Analyze potential traffic violations"""
        violations = {
            'speeding_incidents': [],
            'unsafe_behaviors': [],
            'rule_violations': []
        }
        
        # Group events by type and analyze patterns
        vehicle_events = [e for e in events if any(v in e.event_type.lower() 
                         for v in ['car', 'truck', 'bus', 'motorcycle'])]
        pedestrian_events = [e for e in events if 'person' in e.event_type.lower()]
        
        # Analyze vehicle behavior
        violations['speeding_incidents'] = self._detect_speeding(vehicle_events)
        violations['unsafe_behaviors'] = self._detect_unsafe_behaviors(events)
        violations['rule_violations'] = self._detect_rule_violations(events)
        
        return violations
    
    def _detect_speeding(self, vehicle_events: List[EventDetection]) -> List[Dict[str, Any]]:
        """Detect potential speeding incidents"""
        speeding_incidents = []
        
        # Group vehicles by proximity and time
        for i, event in enumerate(vehicle_events):
            # Look for rapid position changes (simplified approach)
            similar_events = [e for e in vehicle_events 
                            if abs(e.timestamp - event.timestamp) < 2.0 and
                            self._calculate_distance(event.bbox, e.bbox) < 100]
            
            if len(similar_events) > 1:
                # Calculate approximate speed based on position changes
                timestamps = [e.timestamp for e in similar_events]
                positions = [self._get_center(e.bbox) for e in similar_events]
                
                if len(set(timestamps)) > 1:  # Multiple timestamps
                    time_diff = max(timestamps) - min(timestamps)
                    distance = self._calculate_trajectory_distance(positions)
                    
                    # Rough speed estimation (pixels per second)
                    if time_diff > 0:
                        speed = distance / time_diff
                        if speed > 50:  # Threshold for potential speeding
                            speeding_incidents.append({
                                'timestamp': format_timestamp(event.timestamp),
                                'estimated_speed': speed,
                                'confidence': event.confidence,
                                'description': f"Potential speeding detected at {format_timestamp(event.timestamp)}"
                            })
        
        return speeding_incidents[:5]  # Limit to top 5
    
    def _detect_unsafe_behaviors(self, events: List[EventDetection]) -> List[Dict[str, Any]]:
        """Detect unsafe behaviors"""
        unsafe_behaviors = []
        
        # Check for pedestrians and vehicles in close proximity
        for event in events:
            if 'person' in event.event_type.lower():
                # Find nearby vehicles
                nearby_vehicles = [e for e in events 
                                 if abs(e.timestamp - event.timestamp) < 1.0 and
                                 any(v in e.event_type.lower() for v in ['car', 'truck', 'bus']) and
                                 self._calculate_distance(event.bbox, e.bbox) < 150]
                
                if nearby_vehicles:
                    unsafe_behaviors.append({
                        'timestamp': format_timestamp(event.timestamp),
                        'type': 'pedestrian_vehicle_proximity',
                        'description': f"Pedestrian in close proximity to vehicles at {format_timestamp(event.timestamp)}",
                        'confidence': event.confidence
                    })
        
        return unsafe_behaviors[:5]  # Limit to top 5
    
    def _detect_rule_violations(self, events: List[EventDetection]) -> List[Dict[str, Any]]:
        """Detect traffic rule violations"""
        violations = []
        
        # Check for stop sign violations (if stop signs are detected)
        stop_sign_events = [e for e in events if 'stop sign' in e.event_type.lower()]
        vehicle_events = [e for e in events if any(v in e.event_type.lower() 
                         for v in ['car', 'truck', 'bus', 'motorcycle'])]
        
        for stop_sign in stop_sign_events:
            # Find vehicles near stop sign
            nearby_vehicles = [v for v in vehicle_events 
                             if abs(v.timestamp - stop_sign.timestamp) < 2.0 and
                             self._calculate_distance(stop_sign.bbox, v.bbox) < 200]
            
            if nearby_vehicles:
                violations.append({
                    'timestamp': format_timestamp(stop_sign.timestamp),
                    'type': 'stop_sign_area',
                    'description': f"Vehicle activity near stop sign at {format_timestamp(stop_sign.timestamp)}",
                    'confidence': stop_sign.confidence
                })
        
        return violations[:5]  # Limit to top 5
    
    def _calculate_safety_score(self, events: List[EventDetection]) -> float:
        """Calculate overall safety score (0-100)"""
        if not events:
            return 100.0
        
        # Base score
        score = 100.0
        
        # Deduct points for potential violations
        vehicle_count = len([e for e in events if any(v in e.event_type.lower() 
                           for v in ['car', 'truck', 'bus', 'motorcycle'])])
        pedestrian_count = len([e for e in events if 'person' in e.event_type.lower()])
        
        # Deduct points based on activity density
        total_objects = vehicle_count + pedestrian_count
        if total_objects > 50:
            score -= 10  # High traffic density
        elif total_objects > 100:
            score -= 20  # Very high traffic density
        
        # Deduct points for high-risk scenarios
        high_confidence_events = [e for e in events if e.confidence > 0.8]
        risk_factor = len(high_confidence_events) / len(events) if events else 0
        
        if risk_factor > 0.7:
            score -= 15  # High proportion of high-confidence detections may indicate complex scenarios
        
        return max(0.0, min(100.0, score))
    
    def _generate_recommendations(self, events: List[EventDetection]) -> List[str]:
        """Generate safety recommendations based on analysis"""
        recommendations = []
        
        vehicle_count = len([e for e in events if any(v in e.event_type.lower() 
                           for v in ['car', 'truck', 'bus', 'motorcycle'])])
        pedestrian_count = len([e for e in events if 'person' in e.event_type.lower()])
        
        if vehicle_count > 30:
            recommendations.append("High vehicle traffic detected. Consider implementing traffic flow management.")
        
        if pedestrian_count > 20:
            recommendations.append("Significant pedestrian activity observed. Ensure adequate crosswalk signals and safety measures.")
        
        if vehicle_count > 0 and pedestrian_count > 0:
            recommendations.append("Mixed traffic scenario. Monitor for potential conflicts between vehicles and pedestrians.")
        
        # Check for traffic lights
        traffic_light_events = [e for e in events if 'traffic light' in e.event_type.lower()]
        if not traffic_light_events and vehicle_count > 10:
            recommendations.append("No traffic lights detected in high-traffic area. Consider installing traffic control devices.")
        
        if not recommendations:
            recommendations.append("Traffic scenario appears normal. Continue regular monitoring.")
        
        return recommendations
    
    # Helper methods
    def _calculate_distance(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate distance between two bounding boxes"""
        center1 = self._get_center(bbox1)
        center2 = self._get_center(bbox2)
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def _get_center(self, bbox: List[float]) -> tuple:
        """Get center point of bounding box"""
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    
    def _calculate_trajectory_distance(self, positions: List[tuple]) -> float:
        """Calculate total distance of trajectory"""
        if len(positions) < 2:
            return 0
        
        total_distance = 0
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            total_distance += np.sqrt(dx**2 + dy**2)
        
        return total_distance
