"""
Camera Class

A class for handling camera operations using OpenCV.
Provides methods to initialize camera, capture frames, and display video feed.
"""

import cv2
import threading
import time
from hand_detector import HandDetector


class Camera:
    """
    A class to handle camera operations for real-time video capture.
    
    This class provides methods to initialize the camera, capture frames,
    and manage the video feed display.
    """
    
    def __init__(self, camera_index=0, width=640, height=480, enable_hand_detection=True):
        """
        Initialize the camera.
        
        Args:
            camera_index (int): Index of the camera device (default: 0)
            width (int): Frame width in pixels (default: 640)
            height (int): Frame height in pixels (default: 480)
            enable_hand_detection (bool): Enable hand detection (default: True)
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        # Hand detection
        self.enable_hand_detection = enable_hand_detection
        self.hand_detector = None
        self.last_detection_results = None
        self.detection_lock = threading.Lock()
        
        if self.enable_hand_detection:
            try:
                self.hand_detector = HandDetector()
            except Exception as e:
                self.enable_hand_detection = False
                self.hand_detector = None

    
    def start(self):
        """
        Start the camera and begin capturing frames.
        
        Returns:
            bool: True if camera started successfully, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Verify the settings
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

            
            self.is_running = True
            return True
            
        except Exception as e:
            return False
    
    def stop(self):
        """Stop the camera and release resources."""
        try:
            self.is_running = False
            
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            
        except Exception as e:
            pass
    
    def get_frame(self, process_hands=True):
        """
        Get the current frame from the camera, optionally with hand detection.
        
        Args:
            process_hands (bool): Whether to process hands in this frame (default: True)
        
        Returns:
            numpy.ndarray or None: Current frame as BGR image, None if no frame available
        """
        if not self.is_running or self.cap is None:
            return None
        
        try:
            ret, frame = self.cap.read()
            
            if ret:
                with self.frame_lock:
                    self.current_frame = frame.copy()
                
                # Process hands if enabled and requested
                if self.enable_hand_detection and self.hand_detector and process_hands:
                    try:
                        detection_results = self.hand_detector.process_frame(frame)
                        with self.detection_lock:
                            self.last_detection_results = detection_results
                    except Exception as e:
                        pass
                
                return frame
            else:
                return None
                
        except Exception as e:
            return None
    
    def get_current_frame(self):
        """
        Get a copy of the most recent frame (thread-safe).
        
        Returns:
            numpy.ndarray or None: Copy of current frame, None if no frame available
        """
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
            return None
    
    def is_camera_running(self):
        """
        Check if the camera is currently running.
        
        Returns:
            bool: True if camera is running, False otherwise
        """
        return self.is_running and self.cap is not None and self.cap.isOpened()
    
    def get_camera_info(self):
        """
        Get information about the camera.
        
        Returns:
            dict: Dictionary containing camera information
        """
        if not self.is_camera_running():
            return {"status": "Camera not running"}
        
        try:
            info = {
                "status": "Running",
                "index": self.camera_index,
                "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": self.cap.get(cv2.CAP_PROP_FPS),
                "brightness": self.cap.get(cv2.CAP_PROP_BRIGHTNESS),
                "contrast": self.cap.get(cv2.CAP_PROP_CONTRAST),
                "saturation": self.cap.get(cv2.CAP_PROP_SATURATION),
            }
            return info
            
        except Exception as e:
            return {"status": "Error retrieving info"}
    
    def set_resolution(self, width, height):
        """
        Set the camera resolution.
        
        Args:
            width (int): New frame width
            height (int): New frame height
        
        Returns:
            bool: True if resolution set successfully, False otherwise
        """
        if not self.is_camera_running():
            return False
        
        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            # Verify the new settings
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            self.width = actual_width
            self.height = actual_height
            
            return True
            
        except Exception as e:
            return False
    
    def get_hand_detection_results(self):
        """
        Get the latest hand detection results (thread-safe).
        
        Returns:
            dict or None: Latest detection results, None if no results available
        """
        with self.detection_lock:
            if self.last_detection_results is not None:
                return self.last_detection_results.copy()
            return None
    
    def draw_hand_landmarks(self, frame, detection_results=None):
        """
        Draw hand landmarks on a frame.
        
        Args:
            frame (numpy.ndarray): Frame to draw landmarks on
            detection_results (dict, optional): Specific detection results to use
        
        Returns:
            numpy.ndarray: Frame with landmarks drawn
        """
        if not self.enable_hand_detection or self.hand_detector is None:
            return frame
        
        try:
            # Use provided results or get latest results
            if detection_results is None:
                detection_results = self.get_hand_detection_results()
            
            if detection_results is not None:
                return self.hand_detector.draw_landmarks(frame, detection_results)
            
            return frame
            
        except Exception as e:
            return frame
    
    def draw_hand_landmarks_flipped(self, frame, detection_results):
        """
        Draw hand landmarks on a horizontally flipped frame with corrected coordinates.
        
        Args:
            frame (numpy.ndarray): Horizontally flipped frame to draw landmarks on
            detection_results (dict): Detection results from the original (unflipped) frame
        
        Returns:
            numpy.ndarray: Frame with landmarks drawn at flipped positions
        """
        if not self.enable_hand_detection or self.hand_detector is None or not detection_results:
            return frame
        
        try:
            return self.hand_detector.draw_landmarks_flipped(frame, detection_results)
            
        except Exception as e:
            return frame
    
    def get_finger_positions(self):
        """
        Get finger tip positions from the latest detection results.
        
        Returns:
            list: List of finger tip positions for each detected hand
        """
        if not self.enable_hand_detection or self.hand_detector is None:
            return []
        
        detection_results = self.get_hand_detection_results()
        if detection_results is not None:
            return self.hand_detector.get_finger_positions(detection_results)
        
        return []

    
    def is_hands_detected(self):
        """
        Check if hands are currently detected.
        
        Returns:
            bool: True if hands are detected, False otherwise
        """
        detection_results = self.get_hand_detection_results()
        return detection_results is not None and detection_results.get('hands_detected', False)
    

    
    def toggle_hand_detection(self):
        """
        Toggle hand detection on/off.
        
        Returns:
            bool: New state of hand detection (True if enabled, False if disabled)
        """
        if self.hand_detector is None:
            try:
                self.hand_detector = HandDetector()
                self.enable_hand_detection = True
            except Exception as e:
                return False
        else:
            self.enable_hand_detection = not self.enable_hand_detection
        
        return self.enable_hand_detection
    
    def cleanup_hand_detector(self):
        """Clean up hand detector resources."""
        try:
            if self.hand_detector is not None:
                self.hand_detector.cleanup()
                self.hand_detector = None
            pass
        except Exception as e:
            pass
    
    def __del__(self):
        """Destructor to ensure camera resources are released."""
        self.cleanup_hand_detector()
        self.stop()
