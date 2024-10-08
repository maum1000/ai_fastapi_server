# ai_system/core/factories.py

from ..models.FaceDetector import FaceDetectorFactory
from ..models.FacePredictor import FacePredictorFactory

__all__ = [
    "FaceDetectorFactory",
    "FacePredictorFactory",
]
