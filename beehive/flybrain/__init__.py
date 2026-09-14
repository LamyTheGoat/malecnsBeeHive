"""A connectome-flavoured Drosophila brain that learns which way you swipe."""

from .connectome import ConnectomeSource, default_source, load_export, resolve
from .eye import EyeParams, FlyEye
from .fly import DAN_DRIVE, Fly, Percept
from .mushroom_body import MBON_CHANNELS, MBParams, MushroomBody
from .steering import Decision, Steering, SteeringParams

__all__ = [
    "ConnectomeSource", "DAN_DRIVE", "Decision", "EyeParams", "Fly", "FlyEye",
    "MBON_CHANNELS", "MBParams", "MushroomBody", "Percept", "Steering",
    "SteeringParams", "default_source", "load_export", "resolve",
]
__version__ = "0.1.0"
