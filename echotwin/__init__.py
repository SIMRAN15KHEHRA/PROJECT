"""
EchoTwin: AI-powered Digital Twin for Voice Analysis and Simulation

A comprehensive system for:
- Voice disorder detection
- Voice simulation and damage modeling
- Voice health visualization
- Text-to-speech synthesis
"""

__version__ = "1.0.0"
__author__ = "EchoTwin Development Team"

from .audio_processor import AudioProcessor
from .feature_extractor import FeatureExtractor
from .voice_classifier import VoiceClassifier
from .voice_simulator import VoiceSimulator
from .visualization import VoiceVisualizer
from .ui import EchoTwinUI

__all__ = [
    'AudioProcessor',
    'FeatureExtractor', 
    'VoiceClassifier',
    'VoiceSimulator',
    'VoiceVisualizer',
    'EchoTwinUI'
]