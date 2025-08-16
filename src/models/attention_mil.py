"""
Attention MIL 모델 모듈

Multiple Instance Learning과 Attention 메커니즘을 결합한 의료 영상 분류 모델을 정의합니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .feature_extractor import FeatureExtractor


class AttentionMILModel(nn.Module):
    """
    Attention 기반 Multiple Instance Learning 모델
    
    의료 영상의 여러 프레임에서 중요한 부분을 자동으로 식별하고 분류합니다.
    """
    
    def __init__(self, 
                 num_classes: int,
                 feature_extractor: FeatureExtractor,
                 attention_hidden_dim: int = 128,
                 dropout_rate: float = 0.2):
        """
        Attention MIL 모델 초기화
        
        Args:
            num_classes (int): 분류할 클래스 수
            feature_extractor (FeatureExtractor): 특징 추출기
            attention_hidden_dim (int): Attention 레이어의 은닉 차원 (기본값: 128)
            dropout_rate (float): Dropout 비율 (기본값: 0.2)
        """
        super(AttentionMILModel, self).__init__()
        
        self.num_classes = num_classes
        self.feature_extractor = feature_extractor
        self.image_feature_dim = feature_extractor.get_feature_dim()
        
        # Attention 메커니즘
        self.attention = nn.Sequential(
            nn.Linear(self.image_feature_dim, attention_hidden_dim),
            nn.Tanh(),
            nn.Linear(attention_hidden_dim, 1)
        )
        
        # 분류 레이어
        self.classification_layer = nn.Linear(self.image_feature_dim, num_classes)
        
        # Dropout 레이어
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        모델 순전파
        
        Args:
            inputs (torch.Tensor): 입력 텐서 (B, num_tiles, C, H, W)
            
        Returns:
            torch.Tensor: 분류 로짓 (B, num_classes)
        """
        # 입력 차원 검증
        if inputs.dim() != 5:
            raise ValueError(f"입력 텐서는 5차원이어야 합니다. 현재 차원: {inputs.dim()}")
        
        batch_size, num_tiles, channels, height, width = inputs.size()
        
        # 입력을 평면화
        inputs = inputs.view(-1, channels, height, width)  # (B * num_tiles, C, H, W)
        
        # 특징 추출
        features = self.feature_extractor(inputs)  # (B * num_tiles, feature_dim, 1, 1)
        
        # 특징을 재구성
        features = features.view(batch_size, num_tiles, -1)  # (B, num_tiles, feature_dim)
        
        # Attention 메커니즘
        attention_weights = self.attention(features)  # (B, num_tiles, 1)
        attention_weights = F.softmax(attention_weights, dim=1)  # Attention 가중치 정규화
        
        # Attention 가중치를 특징에 적용
        attended_features = torch.sum(features * attention_weights, dim=1)  # (B, feature_dim)
        
        # Dropout 및 활성화 함수 적용
        attended_features = self.dropout(attended_features)
        attended_features = F.relu(attended_features)
        
        # 분류 레이어
        logits = self.classification_layer(attended_features)  # (B, num_classes)
        
        return logits
    
    def get_attention_weights(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Attention 가중치 반환 (시각화용)
        
        Args:
            inputs (torch.Tensor): 입력 텐서 (B, num_tiles, C, H, W)
            
        Returns:
            torch.Tensor: Attention 가중치 (B, num_tiles)
        """
        # 입력 차원 검증
        if inputs.dim() != 5:
            raise ValueError(f"입력 텐서는 5차원이어야 합니다. 현재 차원: {inputs.dim()}")
        
        batch_size, num_tiles, channels, height, width = inputs.size()
        
        # 입력을 평면화
        inputs = inputs.view(-1, channels, height, width)
        
        # 특징 추출
        features = self.feature_extractor(inputs)
        
        # 특징을 재구성
        features = features.view(batch_size, num_tiles, -1)
        
        # Attention 가중치 계산
        attention_weights = self.attention(features)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        return attention_weights.squeeze(-1)  # (B, num_tiles) 