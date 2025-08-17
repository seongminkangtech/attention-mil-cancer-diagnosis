"""
특징 추출기 모듈

의료 영상에서 특징을 추출하는 FeatureExtractor 클래스를 정의합니다.
"""

import timm
import torch
import torch.nn as nn


class FeatureExtractor(nn.Module):
    """
    의료 영상 특징 추출기

    EfficientNet-B2를 기반으로 의료 영상에서 특징을 추출합니다.
    분류 헤드를 제거하고 특징 추출 부분만 사용합니다.
    """

    def __init__(self, model_name: str = "efficientnet_b2", pretrained: bool = True):
        """
        특징 추출기 초기화

        Args:
            model_name (str): 사용할 모델 이름 (기본값: 'efficientnet_b2')
            pretrained (bool): 사전 훈련된 가중치 사용 여부 (기본값: True)
        """
        super(FeatureExtractor, self).__init__()

        # EfficientNet 모델 생성
        self.backbone = timm.create_model(model_name, pretrained=pretrained)

        # 분류 헤드를 제거하고 특징 추출 부분만 사용
        self.feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])

        # 특징 차원 저장
        if hasattr(self.backbone, "num_features"):
            num_features = self.backbone.num_features
            if isinstance(num_features, torch.Tensor):
                self.feature_dim = int(num_features.item())
            elif isinstance(num_features, (int, float)):
                self.feature_dim = int(num_features)
            else:
                # 기본값 설정 (EfficientNet-B2의 특징 차원)
                self.feature_dim = 1408
        else:
            # 기본값 설정 (EfficientNet-B2의 특징 차원)
            self.feature_dim = 1408

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        특징 추출 수행

        Args:
            inputs (torch.Tensor): 입력 이미지 텐서 (B, C, H, W)

        Returns:
            torch.Tensor: 추출된 특징 텐서 (B, feature_dim, 1, 1)
        """
        features = self.feature_extractor(inputs)
        return features

    def get_feature_dim(self) -> int:
        """
        특징 차원 반환

        Returns:
            int: 특징 차원
        """
        return self.feature_dim
