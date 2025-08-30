"""
Gesture Detector Class

A class for real-time gesture recognition using the trained transformer model.
Processes hand landmarks and outputs gesture probabilities.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from collections import deque
from typing import Dict, List, Optional, Tuple
import sys
from pathlib import Path

# Add training directory to path to import model
sys.path.append(str(Path(__file__).parent / 'training'))

from gesture_transformer import GestureTransformer, HandLandmarkProcessor


class GestureDetector:
    """
    Real-time gesture detector using trained transformer model.
    
    This class manages landmark sequences, model inference, and gesture prediction
    for real-time gesture recognition applications.
    """
    
    def __init__(self, model_path="models/best_model.pth", sequence_length=32, confidence_threshold=0.1):
        """
        Initialize the gesture detector.
        
        Args:
            model_path (str): Path to the trained model file
            sequence_length (int): Number of frames to use for prediction
            confidence_threshold (float): Minimum confidence for gesture detection
        """
        self.model_path = model_path
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Gesture classes - mapping from model output indices
        self.gesture_classes = [
            "Left_Swipe",
            "Right_Swipe", 
            "Stop",
            "Thumbs_Down",
            "Thumbs_Up"
        ]
        
        # Initialize landmark processor
        self.landmark_processor = HandLandmarkProcessor()
        
        # Initialize sequence buffer
        self.landmark_sequence = deque(maxlen=self.sequence_length)
        
        # Load model
        self.model = None
        self.load_model()
        
        # Prediction smoothing
        self.prediction_history = deque(maxlen=5)  # Keep last 5 predictions for smoothing
        
    def load_model(self):
        """Load the trained gesture recognition model."""
        try:
            print(f"Loading model from: {self.model_path}")
            
            # Initialize model with correct parameters (matching trained model)
            self.model = GestureTransformer(
                num_classes=5,  # 5 main gesture classes
                d_model=512,  # Based on inspection: input_projection.weight shape [512, 63]
                nhead=8,  # 512 is divisible by 8
                num_encoder_layers=8,  # Based on inspection: found layers 0-7
                dim_feedforward=1024,
                dropout=0.1,
                max_seq_length=64,
                landmark_dim=63
            )
            
            # Load state dict
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            # Move model to device and set to eval mode
            self.model.to(self.device)
            self.model.eval()
            
            print("✅ Model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None
    
    def process_landmarks(self, detection_results):
        """
        Process hand detection results and add to sequence buffer.
        
        Args:
            detection_results (dict): Hand detection results from HandDetector
            
        Returns:
            bool: True if landmarks were successfully processed
        """
        if not detection_results or not detection_results.get('hands_detected', False):
            # Add empty/null frame to maintain temporal consistency
            null_landmarks = np.zeros(63, dtype=np.float32)
            self.landmark_sequence.append(null_landmarks)
            return False
        
        try:
            # Get landmarks from the first detected hand
            hand_landmarks = detection_results['hand_landmarks'][0]
            
            # Convert to feature vector
            landmarks_vector = self.landmark_processor.landmarks_to_vector(hand_landmarks)
            
            # Normalize landmarks relative to wrist
            normalized_landmarks = self.landmark_processor.normalize_landmarks(landmarks_vector)
            
            # Add to sequence buffer
            self.landmark_sequence.append(normalized_landmarks)
            
            return True
            
        except Exception as e:
            print(f"Error processing landmarks: {e}")
            # Add null frame on error
            null_landmarks = np.zeros(63, dtype=np.float32)
            self.landmark_sequence.append(null_landmarks)
            return False
    
    def predict_gesture(self):
        """
        Predict gesture from current landmark sequence.
        
        Returns:
            dict: Dictionary containing gesture probabilities and predicted gesture
        """
        if self.model is None:
            return self._get_empty_prediction()
        
        if len(self.landmark_sequence) < self.sequence_length:
            return self._get_empty_prediction()
        
        try:
            # Prepare input sequence
            sequence = np.array(list(self.landmark_sequence))  # [seq_len, 63]
            
            # Add batch dimension and convert to tensor
            input_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)  # [1, seq_len, 63]
            
            # Create padding mask (all False since we have full sequence)
            padding_mask = torch.zeros(1, self.sequence_length, dtype=torch.bool, device=self.device)
            
            # Predict
            with torch.no_grad():
                logits = self.model(input_tensor, padding_mask)  # [1, num_classes]
                
                # Convert to probabilities
                probabilities = F.softmax(logits, dim=1).cpu().numpy()[0]  # [num_classes]
            
            # Create prediction dict
            prediction = {
                'probabilities': {},
                'predicted_gesture': None,
                'confidence': 0.0,
                'raw_logits': logits.cpu().numpy()[0].tolist()
            }
            
            # Map probabilities to gesture names
            for i, gesture_name in enumerate(self.gesture_classes):
                prediction['probabilities'][gesture_name] = float(probabilities[i])
            
            # Find predicted gesture
            max_prob_idx = np.argmax(probabilities)
            max_prob = probabilities[max_prob_idx]
            
            if max_prob > self.confidence_threshold:
                prediction['predicted_gesture'] = self.gesture_classes[max_prob_idx]
                prediction['confidence'] = float(max_prob)
            
            # Add to prediction history for smoothing
            self.prediction_history.append(prediction)
            
            # Return smoothed prediction
            return self._smooth_predictions()
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            return self._get_empty_prediction()
    
    def _smooth_predictions(self):
        """
        Smooth predictions using recent history.
        
        Returns:
            dict: Smoothed prediction
        """
        if not self.prediction_history:
            return self._get_empty_prediction()
        
        # Average probabilities over recent predictions
        smoothed_probs = {}
        for gesture_name in self.gesture_classes:
            probs = [pred['probabilities'].get(gesture_name, 0.0) for pred in self.prediction_history]
            smoothed_probs[gesture_name] = float(np.mean(probs))
        
        # Find smoothed predicted gesture
        max_gesture = max(smoothed_probs.items(), key=lambda x: x[1])
        
        prediction = {
            'probabilities': smoothed_probs,
            'predicted_gesture': max_gesture[0] if max_gesture[1] > self.confidence_threshold else None,
            'confidence': max_gesture[1],
            'raw_logits': self.prediction_history[-1].get('raw_logits', [0.0] * 5)
        }
        
        return prediction
    
    def _get_empty_prediction(self):
        """
        Get empty prediction when no valid prediction can be made.
        
        Returns:
            dict: Empty prediction
        """
        return {
            'probabilities': {gesture: 0.0 for gesture in self.gesture_classes},
            'predicted_gesture': None,
            'confidence': 0.0,
            'raw_logits': [0.0] * 5
        }
    
    def reset_sequence(self):
        """Reset the landmark sequence buffer."""
        self.landmark_sequence.clear()
        self.prediction_history.clear()
    
    def get_sequence_info(self):
        """
        Get information about current sequence buffer.
        
        Returns:
            dict: Sequence information
        """
        return {
            'sequence_length': len(self.landmark_sequence),
            'max_length': self.sequence_length,
            'is_full': len(self.landmark_sequence) == self.sequence_length,
            'non_null_frames': sum(1 for seq in self.landmark_sequence if np.any(seq != 0))
        }
    
    def draw_predictions(self, frame, prediction):
        """
        Draw gesture predictions on the frame.
        
        Args:
            frame (numpy.ndarray): Frame to draw on
            prediction (dict): Prediction results
            
        Returns:
            numpy.ndarray: Frame with predictions drawn
        """
        if frame is None:
            return frame
        
        try:
            height, width = frame.shape[:2]
            
            # Draw background panel
            panel_width = 300
            panel_height = 200
            panel_x = width - panel_width - 10
            panel_y = 10
            
            # Semi-transparent background
            overlay = frame.copy()
            cv2.rectangle(overlay, (panel_x, panel_y), 
                         (panel_x + panel_width, panel_y + panel_height), 
                         (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Draw title
            cv2.putText(frame, "Gesture Probabilities", 
                       (panel_x + 10, panel_y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw probabilities
            y_offset = panel_y + 50
            for gesture_name in self.gesture_classes:
                prob = prediction['probabilities'].get(gesture_name, 0.0)
                
                # Color based on probability
                if prob > 0.7:
                    color = (0, 255, 0)  # Green for high confidence
                elif prob > 0.3:
                    color = (0, 255, 255)  # Yellow for medium confidence
                else:
                    color = (255, 255, 255)  # White for low confidence
                
                # Draw gesture name and probability
                text = f"{gesture_name}: {prob:.3f}"
                cv2.putText(frame, text, (panel_x + 10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Draw probability bar
                bar_width = int(200 * prob)
                cv2.rectangle(frame, (panel_x + 10, y_offset + 5), 
                             (panel_x + 10 + bar_width, y_offset + 15), 
                             color, -1)
                
                y_offset += 30
            
            # Draw predicted gesture if available
            if prediction['predicted_gesture']:
                cv2.putText(frame, f"Detected: {prediction['predicted_gesture']}", 
                           (panel_x + 10, panel_y + panel_height - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw sequence info
            seq_info = self.get_sequence_info()
            seq_text = f"Buffer: {seq_info['sequence_length']}/{seq_info['max_length']}"
            cv2.putText(frame, seq_text, (10, height - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            return frame
            
        except Exception as e:
            print(f"Error drawing predictions: {e}")
            return frame
    
    def cleanup(self):
        """Clean up resources."""
        try:
            if self.model is not None:
                del self.model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        except Exception as e:
            print(f"Error during cleanup: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()


def main():
    """
    Test function for gesture detector.
    """
    print("🤖 Testing Gesture Detector...")
    
    # Initialize detector
    detector = GestureDetector()
    
    if detector.model is None:
        print("❌ Model not loaded, cannot test")
        return
    
    print("✅ Gesture Detector initialized successfully!")
    print(f"📊 Gesture Classes: {detector.gesture_classes}")
    print(f"🔧 Sequence Length: {detector.sequence_length}")
    print(f"💻 Device: {detector.device}")
    
    # Test with empty prediction
    empty_prediction = detector.predict_gesture()
    print(f"\n📈 Empty prediction: {empty_prediction}")
    
    # Test sequence info
    seq_info = detector.get_sequence_info()
    print(f"📊 Sequence info: {seq_info}")


if __name__ == "__main__":
    main()
