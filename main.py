"""
Main Application Class

Main class that orchestrates the camera and display functionality.
Handles the main application loop and user interactions.
"""

import cv2
import logging
import sys
import time
from camera import Camera
from volume_controller import VolumeController


class MainApp:
    """
    Main application class that manages the camera display and user interactions.
    
    This class coordinates between the Camera and VolumeController classes
    to provide a complete real-time gesture recognition system.
    """
    
    def __init__(self, camera_index=0, window_name="Gesture Recognition Camera"):
        """
        Initialize the main application.
        
        Args:
            camera_index (int): Index of the camera device (default: 0)
            window_name (str): Name of the display window
        """
        self.window_name = window_name
        self.camera = Camera(camera_index=camera_index)
        self.volume_controller = None
        self.is_running = False
        
        # Initialize volume controller
        try:
            self.volume_controller = VolumeController()
            logging.info("Volume controller initialized successfully")
        except Exception as e:
            logging.warning(f"Volume controller initialization failed: {e}")
            self.volume_controller = None
        
        logging.info(f"Main application initialized with camera {camera_index}")
    
    def setup_window(self):
        """Set up the display window with proper settings."""
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_TOPMOST, 1)
            
            # Add some helpful text instructions
            self.create_help_overlay()
            
            logging.info(f"Display window '{self.window_name}' created")
            
        except Exception as e:
            logging.error(f"Failed to setup window: {e}")
            raise
    
    def create_help_overlay(self):
        """Create a help overlay with keyboard shortcuts."""
        self.help_text = [
            "Gesture Recognition Camera",
            "",
            "Controls:",
            "ESC or Q - Quit application",
            "SPACE - Show/hide this help",
            "I - Show camera info",
            "R - Reset camera",
            "H - Toggle hand detection",
            "",
            "Volume Controls (if available):",
            "↑ - Increase volume",
            "↓ - Decrease volume",
            "M - Toggle mute",
        ]
    
    def draw_help_overlay(self, frame):
        """
        Draw help text overlay on the frame.
        
        Args:
            frame (numpy.ndarray): Frame to draw overlay on
        
        Returns:
            numpy.ndarray: Frame with overlay
        """
        overlay = frame.copy()
        
        # Semi-transparent background
        cv2.rectangle(overlay, (10, 10), (400, 320), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw help text
        y_offset = 30
        for line in self.help_text:
            cv2.putText(frame, line, (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
        
        return frame
    
    def draw_status_info(self, frame):
        """
        Draw status information on the frame.
        
        Args:
            frame (numpy.ndarray): Frame to draw status on
        
        Returns:
            numpy.ndarray: Frame with status info
        """
        height, width = frame.shape[:2]
        
        # Draw FPS (placeholder for now)
        cv2.putText(frame, "FPS: 30", (width - 100, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw hand detection status
        if self.camera.enable_hand_detection:
            cv2.putText(frame, "Hand Detection: ON", (width - 200, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Hand Detection: OFF", (width - 200, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Draw volume info if volume controller is available
        if self.volume_controller:
            try:
                volume = self.volume_controller.get_current_volume()
                muted = self.volume_controller.is_muted()
                
                volume_text = f"Volume: {volume:.0f}%"
                if muted:
                    volume_text += " (MUTED)"
                
                cv2.putText(frame, volume_text, (width - 200, height - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            except Exception as e:
                logging.error(f"Error displaying volume info: {e}")
        
        return frame
    
    def handle_keyboard_input(self, key):
        """
        Handle keyboard input for controlling the application.
        
        Args:
            key (int): Key code from cv2.waitKey()
        
        Returns:
            bool: True if application should continue, False to quit
        """
        key = key & 0xFF
        
        # Quit application
        if key == 27 or key == ord('q'):  # ESC or 'q'
            return False
        
        # Show camera info
        elif key == ord('i'):
            info = self.camera.get_camera_info()
            logging.info(f"Camera info: {info}")
            print("Camera Info:")
            for k, v in info.items():
                print(f"  {k}: {v}")
        
        # Reset camera
        elif key == ord('r'):
            logging.info("Resetting camera...")
            self.camera.stop()
            time.sleep(0.5)
            if self.camera.start():
                logging.info("Camera reset successfully")
            else:
                logging.error("Failed to reset camera")
        
        # Toggle hand detection
        elif key == ord('h'):
            enabled = self.camera.toggle_hand_detection()
            logging.info(f"Hand detection {'enabled' if enabled else 'disabled'}")
        
        # Volume controls (if volume controller is available)
        elif self.volume_controller:
            if key == 82:  # Up arrow key
                new_volume = self.volume_controller.increase_volume(5)
                logging.info(f"Volume increased to {new_volume:.1f}%")
            
            elif key == 84:  # Down arrow key
                new_volume = self.volume_controller.decrease_volume(5)
                logging.info(f"Volume decreased to {new_volume:.1f}%")
            
            elif key == ord('m'):  # Toggle mute
                if self.volume_controller.is_muted():
                    self.volume_controller.unmute()
                else:
                    self.volume_controller.mute()
        
        return True
    
    def run(self):
        """
        Main application loop.
        
        Returns:
            int: Exit code (0 for success, 1 for error)
        """
        try:
            # Setup logging
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s'
            )
            
            logging.info("Starting gesture recognition application...")
            
            # Initialize camera
            if not self.camera.start():
                logging.error("Failed to start camera")
                return 1
            
            # Setup display window
            self.setup_window()
            
            self.is_running = True
            show_help = False
            
            logging.info("Application started successfully. Press ESC or 'q' to quit.")
            
            # Main loop
            while self.is_running:
                # Get frame from camera
                frame = self.camera.get_frame()
                
                if frame is None:
                    logging.warning("No frame received from camera")
                    time.sleep(0.1)
                    continue
                                
                # Process hands before flipping
                detection_results = None
                if self.camera.enable_hand_detection:
                    detection_results = self.camera.get_hand_detection_results()
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Draw hand landmarks if hand detection is enabled (with flipped coordinates)
                if self.camera.enable_hand_detection and detection_results:
                    frame = self.camera.draw_hand_landmarks_flipped(frame, detection_results)
                
                # Add status information
                frame = self.draw_status_info(frame)
                
                # Show help overlay if requested
                if show_help:
                    frame = self.draw_help_overlay(frame)
                
                # Display the frame
                cv2.imshow(self.window_name, frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1)
                if key != -1:
                    if key == ord(' '):  # Space key toggles help
                        show_help = not show_help
                    elif not self.handle_keyboard_input(key):
                        break
            
            logging.info("Application shutting down...")
            return 0
            
        except KeyboardInterrupt:
            logging.info("Application interrupted by user")
            return 0
            
        except Exception as e:
            logging.error(f"Application error: {e}")
            return 1
            
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources before application exit."""
        try:
            self.is_running = False
            
            # Stop camera
            self.camera.stop()
            
            # Close OpenCV windows
            cv2.destroyAllWindows()
            
            logging.info("Cleanup completed")
            
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()


def main():
    """Entry point for the application."""
    app = MainApp()
    exit_code = app.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
