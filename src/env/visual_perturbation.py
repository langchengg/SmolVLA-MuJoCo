"""
Visual Perturbation Engine for VLA Robustness Analysis.

Implements systematic visual perturbations (brightness, contrast, noise,
camera shifts, background changes) for evaluating VLA model robustness.
This is a key innovation point of the project.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class PerturbationType(Enum):
    BRIGHTNESS = "brightness"
    CONTRAST = "contrast"
    GAUSSIAN_NOISE = "gaussian_noise"
    CAMERA_SHIFT = "camera_shift"
    BACKGROUND_CHANGE = "background_change"
    CAMERA_ANGLE = "camera_angle"
    COLOR_JITTER = "color_jitter"
    MOTION_BLUR = "motion_blur"
    OCCLUSION = "occlusion"


@dataclass
class PerturbationConfig:
    """Configuration for a specific perturbation."""
    ptype: PerturbationType
    level: float = 1.0     # Perturbation intensity (1.0 = nominal)
    params: dict = None    # Additional parameters
    
    def __post_init__(self):
        if self.params is None:
            self.params = {}


class VisualPerturbationEngine:
    """
    Engine for applying systematic visual perturbations to observations.
    
    Used for Innovation Point #2: Visual Perturbation Robustness Analysis.
    Supports composable perturbations for comprehensive robustness evaluation.
    
    Example:
        >>> engine = VisualPerturbationEngine()
        >>> engine.add_perturbation(PerturbationType.BRIGHTNESS, level=1.3)
        >>> engine.add_perturbation(PerturbationType.GAUSSIAN_NOISE, level=0.05)
        >>> perturbed_image = engine.apply(original_image)
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self._perturbations: list[PerturbationConfig] = []
        self._background_textures = {}

    def add_perturbation(self, ptype: PerturbationType, level: float = 1.0, **kwargs):
        """Add a perturbation to the pipeline."""
        self._perturbations.append(PerturbationConfig(
            ptype=ptype, level=level, params=kwargs
        ))
        return self  # Allow chaining

    def clear(self):
        """Clear all perturbations."""
        self._perturbations.clear()
        return self

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply all configured perturbations to an image.
        
        Args:
            image: (H, W, 3) uint8 RGB image
            
        Returns:
            Perturbed image, same shape and dtype
        """
        result = image.astype(np.float32) / 255.0  # Normalize to [0, 1]

        for perturbation in self._perturbations:
            result = self._apply_single(result, perturbation)

        # Clip and convert back
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        return result

    def _apply_single(self, image: np.ndarray, config: PerturbationConfig) -> np.ndarray:
        """Apply a single perturbation to a normalized float image."""
        dispatch = {
            PerturbationType.BRIGHTNESS: self._apply_brightness,
            PerturbationType.CONTRAST: self._apply_contrast,
            PerturbationType.GAUSSIAN_NOISE: self._apply_gaussian_noise,
            PerturbationType.CAMERA_SHIFT: self._apply_camera_shift,
            PerturbationType.BACKGROUND_CHANGE: self._apply_background_change,
            PerturbationType.COLOR_JITTER: self._apply_color_jitter,
            PerturbationType.MOTION_BLUR: self._apply_motion_blur,
            PerturbationType.OCCLUSION: self._apply_occlusion,
        }
        
        handler = dispatch.get(config.ptype)
        if handler:
            return handler(image, config)
        
        logger.warning(f"Unknown perturbation type: {config.ptype}")
        return image

    def _apply_brightness(self, image: np.ndarray, config: PerturbationConfig) -> np.ndarray:
        """Adjust brightness by multiplicative factor."""
        return image * config.level

    def _apply_contrast(self, image: np.ndarray, config: PerturbationConfig) -> np.ndarray:
        """Adjust contrast around the mean."""
        mean = image.mean()
        return (image - mean) * config.level + mean

    def _apply_gaussian_noise(self, image: np.ndarray, config: PerturbationConfig) -> np.ndarray:
        """Add Gaussian noise with given sigma level."""
        noise = self.rng.randn(*image.shape).astype(np.float32) * config.level
        return image + noise

    def _apply_camera_shift(self, image: np.ndarray, config: PerturbationConfig) -> np.ndarray:
        """Simulate camera viewpoint shift via affine transform."""
        h, w = image.shape[:2]
        dx = int(config.params.get("x_offset", 0) * w)
        dy = int(config.params.get("y_offset", 0) * h)
        
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(
            (image * 255).astype(np.uint8), M, (w, h),
            borderMode=cv2.BORDER_REPLICATE
        )
        return shifted.astype(np.float32) / 255.0

    def _apply_background_change(self, image: np.ndarray, config: PerturbationConfig) -> np.ndarray:
        """
        Apply background texture change.
        In MuJoCo, this is done at the environment level.
        Here we simulate via a simple overlay for non-MuJoCo images.
        """
        texture_type = config.params.get("texture", "default")
        
        if texture_type == "default":
            return image
        
        h, w = image.shape[:2]
        
        if texture_type == "checkered":
            bg = self._generate_checkered(h, w)
        elif texture_type == "plain_white":
            bg = np.ones((h, w, 3), dtype=np.float32) * 0.9
        elif texture_type == "wood":
            bg = self._generate_wood_texture(h, w)
        elif texture_type == "marble":
            bg = self._generate_marble_texture(h, w)
        else:
            return image
        
        # Simple background replacement: assume dark pixels are background
        gray = np.mean(image, axis=2)
        mask = (gray < 0.15).astype(np.float32)[..., np.newaxis]
        
        result = image * (1 - mask) + bg * mask
        return result

    def _apply_color_jitter(self, image: np.ndarray, config: PerturbationConfig) -> np.ndarray:
        """Random color jitter (hue, saturation, brightness)."""
        hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        
        jitter = config.level
        hsv[:, :, 0] = (hsv[:, :, 0] + self.rng.uniform(-10 * jitter, 10 * jitter)) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + self.rng.uniform(-0.3 * jitter, 0.3 * jitter)), 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1 + self.rng.uniform(-0.2 * jitter, 0.2 * jitter)), 0, 255)
        
        rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return rgb.astype(np.float32) / 255.0

    def _apply_motion_blur(self, image: np.ndarray, config: PerturbationConfig) -> np.ndarray:
        """Apply motion blur to simulate fast camera movement."""
        kernel_size = max(3, int(config.level * 15))
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        kernel[kernel_size // 2, :] = 1.0 / kernel_size
        
        blurred = cv2.filter2D(
            (image * 255).astype(np.uint8), -1, kernel
        )
        return blurred.astype(np.float32) / 255.0

    def _apply_occlusion(self, image: np.ndarray, config: PerturbationConfig) -> np.ndarray:
        """Add random rectangular occlusion patches."""
        h, w = image.shape[:2]
        result = image.copy()
        
        n_patches = int(config.level * 3) + 1
        for _ in range(n_patches):
            patch_h = self.rng.randint(h // 8, h // 4)
            patch_w = self.rng.randint(w // 8, w // 4)
            y = self.rng.randint(0, h - patch_h)
            x = self.rng.randint(0, w - patch_w)
            result[y:y+patch_h, x:x+patch_w] = self.rng.uniform(0.3, 0.7, size=(patch_h, patch_w, 3))
        
        return result

    def _generate_checkered(self, h: int, w: int, tile_size: int = 32) -> np.ndarray:
        """Generate a checkered pattern."""
        pattern = np.zeros((h, w, 3), dtype=np.float32)
        for i in range(0, h, tile_size):
            for j in range(0, w, tile_size):
                if ((i // tile_size) + (j // tile_size)) % 2 == 0:
                    pattern[i:i+tile_size, j:j+tile_size] = 0.8
                else:
                    pattern[i:i+tile_size, j:j+tile_size] = 0.4
        return pattern

    def _generate_wood_texture(self, h: int, w: int) -> np.ndarray:
        """Generate a simple procedural wood-like texture."""
        x = np.linspace(0, 10, w)
        y = np.linspace(0, 10, h)
        xx, yy = np.meshgrid(x, y)
        
        r = np.sqrt(xx**2 + yy**2)
        wood = (np.sin(r * 5 + self.rng.randn(h, w) * 0.5) + 1) / 2
        
        texture = np.stack([
            wood * 0.6 + 0.2,
            wood * 0.4 + 0.15,
            wood * 0.2 + 0.1,
        ], axis=-1).astype(np.float32)
        
        return texture

    def _generate_marble_texture(self, h: int, w: int) -> np.ndarray:
        """Generate a simple procedural marble-like texture."""
        noise = self.rng.randn(h, w).astype(np.float32)
        
        # Smooth the noise
        from scipy.ndimage import gaussian_filter
        smooth = gaussian_filter(noise, sigma=5)
        
        marble = (np.sin(smooth * 3) + 1) / 2
        
        texture = np.stack([
            marble * 0.3 + 0.6,
            marble * 0.3 + 0.6,
            marble * 0.35 + 0.6,
        ], axis=-1).astype(np.float32)
        
        return texture

    @staticmethod
    def get_all_perturbation_configs() -> list[dict]:
        """
        Generate all perturbation configurations for systematic robustness testing.
        Returns a list of dicts, each defining a perturbation condition.
        """
        configs = []
        
        # Brightness variations
        for level in [0.5, 0.75, 1.0, 1.25, 1.5]:
            configs.append({
                "name": f"brightness_{level}",
                "type": PerturbationType.BRIGHTNESS,
                "level": level,
            })
        
        # Contrast variations
        for level in [0.5, 0.75, 1.0, 1.25, 1.5]:
            configs.append({
                "name": f"contrast_{level}",
                "type": PerturbationType.CONTRAST,
                "level": level,
            })
        
        # Gaussian noise
        for sigma in [0.0, 0.02, 0.05, 0.1, 0.15]:
            configs.append({
                "name": f"noise_{sigma}",
                "type": PerturbationType.GAUSSIAN_NOISE,
                "level": sigma,
            })
        
        # Motion blur
        for level in [0.0, 0.3, 0.5, 0.7, 1.0]:
            configs.append({
                "name": f"motion_blur_{level}",
                "type": PerturbationType.MOTION_BLUR,
                "level": level,
            })
        
        return configs

    def __repr__(self):
        names = [p.ptype.value for p in self._perturbations]
        return f"VisualPerturbationEngine(perturbations={names})"
