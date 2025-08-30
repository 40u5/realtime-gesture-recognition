"""
Hand Detector Class

A class for detecting hands and extracting landmarks using MediaPipe.
Provides methods to process frames and get hand detection results.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Optional, Dict, Any


class HandDetector:
    """
    A class to detect hands and extract landmarks using MediaPipe.
    
    This class provides methods to process frames, detect hands,
    and extract hand landmarks for gesture recognition.
    """
    
    def __init__(self, 
                 max_num_hands=2,
                 min_detection_confidence=0.7,
                 min_tracking_confidence=0.5,
                 model_complexity=1):
        """
        Initialize the hand detector.
        
        Args:
            max_num_hands (int): Maximum number of hands to detect (default: 2)
            min_detection_confidence (float): Minimum confidence for hand detection (default: 0.7)
            min_tracking_confidence (float): Minimum confidence for hand tracking (default: 0.5)
            model_complexity (int): Model complexity (0 or 1, default: 1)
        """
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_complexity = model_complexity
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize hands detector
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.max_num_hands,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
            model_complexity=self.model_complexity
        )
        
        # Store the last detection results
        self.last_results = None
        self.hands_detected = False

    
    def process_frame(self, frame):
        """
        Process a frame to detect hands and extract landmarks.
        
        Args:
            frame (numpy.ndarray): Input frame in BGR format
            
        Returns:
            dict: Detection results containing hands information
        """
        if frame is None:
            return None
        
        try:
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            results = self.hands.process(rgb_frame)
            
            # Store results
            self.last_results = results
            self.hands_detected = results.multi_hand_landmarks is not None
            
            # Create detection results dictionary
            detection_results = {
                'hands_detected': self.hands_detected,
                'num_hands': len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0,
                'hand_landmarks': [],
                'hand_classifications': [],
                'frame_shape': frame.shape
            }
            
            if results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Extract landmark coordinates
                    landmarks = self._extract_landmarks(hand_landmarks, frame.shape)
                    detection_results['hand_landmarks'].append(landmarks)
                    
                    # Get hand classification (Left/Right) if available
                    if results.multi_handedness and idx < len(results.multi_handedness):
                        classification = results.multi_handedness[idx].classification[0]
                        hand_info = {
                            'label': classification.label,  # 'Left' or 'Right'
                            'score': classification.score
                        }
                        detection_results['hand_classifications'].append(hand_info)
            
            return detection_results
            
        except Exception as e:
            return None
    
    def _extract_landmarks(self, hand_landmarks, frame_shape):
        """
        Extract landmark coordinates from MediaPipe landmarks.
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            frame_shape: Shape of the frame (height, width, channels)
            
        Returns:
            dict: Dictionary containing landmark coordinates
        """
        height, width = frame_shape[:2]
        landmarks = {}
        
        # Extract all 21 hand landmarks
        landmark_names = [
            'WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP',
            'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP',
            'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP',
            'RING_FINGER_MCP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP',
            'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP'
        ]
        
        for idx, landmark in enumerate(hand_landmarks.landmark):
            if idx < len(landmark_names):
                landmarks[landmark_names[idx]] = {
                    'x': int(landmark.x * width),
                    'y': int(landmark.y * height),
                    'z': landmark.z,  # Depth relative to wrist
                    'normalized_x': landmark.x,
                    'normalized_y': landmark.y
                }
        
        return landmarks
    
    def draw_landmarks(self, frame, detection_results=None):
        """
        Draw hand landmarks on the frame.
        
        Args:
            frame (numpy.ndarray): Frame to draw landmarks on
            detection_results (dict, optional): Detection results from process_frame
            
        Returns:
            numpy.ndarray: Frame with landmarks drawn
        """
        if frame is None:
            return frame
        
        try:
            # Use provided results or last stored results
            results = self.last_results if detection_results is None else None
            
            if detection_results and detection_results['hands_detected']:
                # Draw landmarks using stored coordinates
                for hand_landmarks in detection_results['hand_landmarks']:
                    self._draw_landmark_points(frame, hand_landmarks)
                    self._draw_connections(frame, hand_landmarks)
            
            elif results and results.multi_hand_landmarks:
                # Draw landmarks using MediaPipe's drawing utilities
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
            
            return frame
            
        except Exception as e:
            return frame
    
    def draw_landmarks_flipped(self, frame, detection_results):
        """
        Draw hand landmarks on a horizontally flipped frame with corrected coordinates.
        
        Args:
            frame (numpy.ndarray): Horizontally flipped frame to draw landmarks on
            detection_results (dict): Detection results from the original (unflipped) frame
        
        Returns:
            numpy.ndarray: Frame with landmarks drawn at flipped positions
        """
        if frame is None or not detection_results:
            return frame
        
        try:
            # Create a copy of detection results with flipped coordinates
            flipped_results = detection_results.copy()
            frame_width = frame.shape[1]
            
            if flipped_results['hands_detected'] and flipped_results['hand_landmarks']:
                flipped_hand_landmarks = []
                
                for hand_landmarks in flipped_results['hand_landmarks']:
                    flipped_landmarks = {}
                    
                    # Flip the x coordinates for each landmark
                    for landmark_name, coords in hand_landmarks.items():
                        flipped_landmarks[landmark_name] = coords.copy()
                        # Flip x coordinate: new_x = frame_width - original_x
                        flipped_landmarks[landmark_name]['x'] = frame_width - coords['x']
                        # Update normalized coordinates too
                        flipped_landmarks[landmark_name]['normalized_x'] = 1.0 - coords['normalized_x']
                    
                    flipped_hand_landmarks.append(flipped_landmarks)
                
                flipped_results['hand_landmarks'] = flipped_hand_landmarks
            
            # Draw landmarks with flipped coordinates
            return self.draw_landmarks(frame, flipped_results)
            
        except Exception as e:
            return frame
    
    def _draw_landmark_points(self, frame, landmarks):
        """Draw landmark points on the frame."""
        for landmark_name, coords in landmarks.items():
            x, y = coords['x'], coords['y']
            
            # Different colors for different parts of the hand
            if 'THUMB' in landmark_name:
                color = (255, 0, 0)  # Red for thumb
            elif 'INDEX' in landmark_name:
                color = (0, 255, 0)  # Green for index
            elif 'MIDDLE' in landmark_name:
                color = (0, 0, 255)  # Blue for middle
            elif 'RING' in landmark_name:
                color = (255, 255, 0)  # Cyan for ring
            elif 'PINKY' in landmark_name:
                color = (255, 0, 255)  # Magenta for pinky
            else:
                color = (255, 255, 255)  # White for wrist
            
            cv2.circle(frame, (x, y), 5, color, -1)
    
    def _draw_connections(self, frame, landmarks):
        """Draw connections between landmarks."""
        connections = [
            # Thumb
            ('WRIST', 'THUMB_CMC'), ('THUMB_CMC', 'THUMB_MCP'),
            ('THUMB_MCP', 'THUMB_IP'), ('THUMB_IP', 'THUMB_TIP'),
            
            # Index finger
            ('WRIST', 'INDEX_FINGER_MCP'), ('INDEX_FINGER_MCP', 'INDEX_FINGER_PIP'),
            ('INDEX_FINGER_PIP', 'INDEX_FINGER_DIP'), ('INDEX_FINGER_DIP', 'INDEX_FINGER_TIP'),
            
            # Middle finger
            ('WRIST', 'MIDDLE_FINGER_MCP'), ('MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP'),
            ('MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP'), ('MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP'),
            
            # Ring finger
            ('WRIST', 'RING_FINGER_MCP'), ('RING_FINGER_MCP', 'RING_FINGER_PIP'),
            ('RING_FINGER_PIP', 'RING_FINGER_DIP'), ('RING_FINGER_DIP', 'RING_FINGER_TIP'),
            
            # Pinky
            ('WRIST', 'PINKY_MCP'), ('PINKY_MCP', 'PINKY_PIP'),
            ('PINKY_PIP', 'PINKY_DIP'), ('PINKY_DIP', 'PINKY_TIP'),
            
            # Palm connections
            ('INDEX_FINGER_MCP', 'MIDDLE_FINGER_MCP'),
            ('MIDDLE_FINGER_MCP', 'RING_FINGER_MCP'),
            ('RING_FINGER_MCP', 'PINKY_MCP')
        ]
        
        for start, end in connections:
            if start in landmarks and end in landmarks:
                start_pos = (landmarks[start]['x'], landmarks[start]['y'])
                end_pos = (landmarks[end]['x'], landmarks[end]['y'])
                cv2.line(frame, start_pos, end_pos, (0, 255, 0), 2)
    
    def get_finger_positions(self, detection_results):
        """
        Get finger tip positions for gesture recognition.
        
        Args:
            detection_results (dict): Detection results from process_frame
            
        Returns:
            list: List of finger tip positions for each detected hand
        """
        if not detection_results or not detection_results['hands_detected']:
            return []
        
        finger_tips = ['THUMB_TIP', 'INDEX_FINGER_TIP', 'MIDDLE_FINGER_TIP', 
                      'RING_FINGER_TIP', 'PINKY_TIP']
        
        hands_finger_positions = []
        
        for hand_landmarks in detection_results['hand_landmarks']:
            finger_positions = {}
            for tip in finger_tips:
                if tip in hand_landmarks:
                    finger_positions[tip] = {
                        'x': hand_landmarks[tip]['x'],
                        'y': hand_landmarks[tip]['y'],
                        'normalized_x': hand_landmarks[tip]['normalized_x'],
                        'normalized_y': hand_landmarks[tip]['normalized_y']
                    }
            hands_finger_positions.append(finger_positions)
        
        return hands_finger_positions
    
    def calculate_distance(self, point1, point2):
        """
        Calculate Euclidean distance between two points.
        
        Args:
            point1 (dict): First point with 'x' and 'y' keys
            point2 (dict): Second point with 'x' and 'y' keys
            
        Returns:
            float: Distance between the points
        """
        return np.sqrt((point1['x'] - point2['x'])**2 + (point1['y'] - point2['y'])**2)
    

    
    def get_hand_center(self, landmarks):
        """
        Calculate the center point of a hand based on landmarks.
        
        Args:
            landmarks (dict): Hand landmarks
            
        Returns:
            dict: Center point with 'x' and 'y' coordinates
        """
        if not landmarks:
            return None
        
        x_coords = [landmark['x'] for landmark in landmarks.values()]
        y_coords = [landmark['y'] for landmark in landmarks.values()]
        
        center_x = sum(x_coords) // len(x_coords)
        center_y = sum(y_coords) // len(y_coords)
        
        return {'x': center_x, 'y': center_y}
    
    def cleanup(self):
        """Clean up MediaPipe resources."""
        try:
            if self.hands:
                self.hands.close()
            pass
        except Exception as e:
            pass
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
