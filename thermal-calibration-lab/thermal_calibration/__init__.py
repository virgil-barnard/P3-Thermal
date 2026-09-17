"""Known synthetic geometry and acquisition, not a calibrated model of a real P3."""

from .config import Acquisition, Camera, Scenario, Target
from .dataset import generate_dataset, iter_inputs, load_truth_frame

__all__ = [
    "Acquisition",
    "Camera",
    "Scenario",
    "Target",
    "generate_dataset",
    "iter_inputs",
    "load_truth_frame",
]
__version__ = "0.1.0"
