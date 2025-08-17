"""
모델 설정 패키지

AI 모델별 설정을 dataclass로 관리합니다.
"""

from .attention_mil_config import (AttentionConfig, AttentionMILConfig,
                                   FeatureExtractorConfig)

__all__ = ["FeatureExtractorConfig", "AttentionConfig", "AttentionMILConfig"]
