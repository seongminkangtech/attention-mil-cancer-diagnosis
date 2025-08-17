"""
의료 AI 모델 패키지

이 패키지는 의료 영상 분류를 위한 다양한 모델을 포함합니다.
"""

from .attention_mil import AttentionMILModel
from .feature_extractor import FeatureExtractor

__all__ = ["AttentionMILModel", "FeatureExtractor"]
