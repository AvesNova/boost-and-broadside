import os
import pygame
import numpy as np
from PIL import Image

class PygameWebBridge:
    """
    Bridge to capture Pygame frames for web rendering (Streamlit).
    """

    @staticmethod
    def setup_headless():
        """Configure SDL to run in dummy mode for headless frame capture."""
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["HEADLESS"] = "1"

    @staticmethod
    def surface_to_image(surface: pygame.Surface) -> Image.Image:
        """
        Convert a Pygame surface to a PIL Image.
        
        Args:
            surface: The Pygame surface to convert.
            
        Returns:
            A PIL Image representation of the surface.
        """
        # Get raw data from surface
        data = pygame.image.tostring(surface, "RGB")
        # Create PIL Image
        img = Image.frombytes("RGB", surface.get_size(), data)
        return img

    @staticmethod
    def surface_to_array(surface: pygame.Surface) -> np.ndarray:
        """
        Convert a Pygame surface to a numpy array.
        
        Args:
            surface: The Pygame surface to convert.
            
        Returns:
            A numpy array (H, W, 3) in RGB format.
        """
        # pygame.surfarray.array3d returns (W, H, 3)
        # We need (H, W, 3) for most image processing libraries
        array = pygame.surfarray.array3d(surface)
        return np.transpose(array, (1, 0, 2))
