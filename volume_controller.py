"""
Volume Controller Class

A basic class for controlling system volume on Windows using pycaw.
Provides methods to increase and decrease volume by relative amounts.
"""

from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
from ctypes import cast, POINTER
import logging


class VolumeController:
    """
    A class to control system volume with relative adjustments.
    
    This class uses the Windows Core Audio API (pycaw) to control
    the master volume of the default audio device.
    """
    
    def __init__(self):
        """Initialize the volume controller."""
        try:
            # Get the default audio device
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(
                IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            self.volume = cast(interface, POINTER(IAudioEndpointVolume))
            
            # Get volume range (typically -65.25 to 0.0 in dB)
            self.volume_range = self.volume.GetVolumeRange()
            self.min_volume = self.volume_range[0]
            self.max_volume = self.volume_range[1]
            
            logging.info(f"Volume controller initialized. Range: {self.min_volume:.2f} to {self.max_volume:.2f} dB")
            
        except Exception as e:
            logging.error(f"Failed to initialize volume controller: {e}")
            raise
    
    def get_current_volume(self):
        """
        Get the current volume level as a percentage (0-100).
        
        Returns:
            float: Current volume as percentage (0.0 to 100.0)
        """
        try:
            # Get current volume as scalar (0.0 to 1.0) - matches system volume display
            scalar_volume = self.volume.GetMasterVolumeLevelScalar()
            
            # Convert to percentage (0-100)
            volume_percentage = scalar_volume * 100
            
            return max(0.0, min(100.0, volume_percentage))
            
        except Exception as e:
            logging.error(f"Failed to get current volume: {e}")
            return 0.0
    
    def set_volume_percentage(self, percentage):
        """
        Set the volume to a specific percentage (0-100).
        
        Args:
            percentage (float): Target volume percentage (0.0 to 100.0)
        """
        try:
            # Clamp percentage to valid range
            percentage = max(0.0, min(100.0, percentage))
            
            # Convert percentage to scalar (0.0 to 1.0)
            scalar_volume = percentage / 100.0
            
            # Set the volume using scalar method - matches system volume behavior
            self.volume.SetMasterVolumeLevelScalar(scalar_volume, None)
            
            logging.info(f"Volume set to {percentage:.1f}%")
            
        except Exception as e:
            logging.error(f"Failed to set volume to {percentage}%: {e}")
    
    def increase_volume(self, amount=5):
        """
        Increase the volume by an absolute amount.
        
        Args:
            amount (int/float): Amount to increase volume by (default: 5 units)
        
        Returns:
            float: New volume level after increase
        """
        try:
            current_volume = self.get_current_volume()
            new_volume = current_volume + amount
            
            # Ensure we don't exceed maximum
            new_volume = min(100.0, new_volume)
            
            self.set_volume_percentage(new_volume)
            
            logging.info(f"Volume increased by {amount} units (from {current_volume:.1f} to {new_volume:.1f})")
            
            return new_volume
            
        except Exception as e:
            logging.error(f"Failed to increase volume: {e}")
            return self.get_current_volume()
    
    def decrease_volume(self, amount=5):
        """
        Decrease the volume by an absolute amount.
        
        Args:
            amount (int/float): Amount to decrease volume by (default: 5 units)
        
        Returns:
            float: New volume level after decrease
        """
        try:
            current_volume = self.get_current_volume()
            new_volume = current_volume - amount
            
            # Ensure we don't go below minimum
            new_volume = max(0.0, new_volume)
            
            self.set_volume_percentage(new_volume)
            
            logging.info(f"Volume decreased by {amount} units (from {current_volume:.1f} to {new_volume:.1f})")
            
            return new_volume
            
        except Exception as e:
            logging.error(f"Failed to decrease volume: {e}")
            return self.get_current_volume()
    
    def mute(self):
        """Mute the audio."""
        try:
            self.volume.SetMute(1, None)
            logging.info("Audio muted")
        except Exception as e:
            logging.error(f"Failed to mute audio: {e}")
    
    def unmute(self):
        """Unmute the audio."""
        try:
            self.volume.SetMute(0, None)
            logging.info("Audio unmuted")
        except Exception as e:
            logging.error(f"Failed to unmute audio: {e}")
    
    def is_muted(self):
        """
        Check if audio is currently muted.
        
        Returns:
            bool: True if muted, False otherwise
        """
        try:
            return bool(self.volume.GetMute())
        except Exception as e:
            logging.error(f"Failed to check mute status: {e}")
            return False