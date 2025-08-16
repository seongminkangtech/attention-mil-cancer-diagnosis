"""
유틸리티 함수 단위 테스트 - 실제 존재하는 클래스들만 테스트
"""
import pytest
import torch
from unittest.mock import Mock, patch

from src.utils.dataset import CustomDataset


class TestCustomDataset:
    """CustomDataset 클래스 테스트"""
    
    def test_dataset_initialization(self):
        """데이터셋 초기화 테스트"""
        # 가상의 데이터 정보
        mock_ids = ['sample1', 'sample2']
        mock_image_tensors = torch.randn(2, 10, 3, 256, 256)  # (N, num_frames, C, H, W)
        mock_labels = torch.tensor([[1, 0, 0], [0, 1, 0]])  # one-hot encoding
        
        dataset = CustomDataset(
            ids=mock_ids,
            image_tensors=mock_image_tensors,
            labels=mock_labels
        )
        
        assert len(dataset) == 2
        assert dataset.ids == mock_ids
    
    def test_dataset_getitem(self):
        """데이터셋 아이템 접근 테스트"""
        mock_ids = ['sample1']
        mock_image_tensors = torch.randn(1, 10, 3, 256, 256)
        mock_labels = torch.tensor([[1, 0, 0]])
        
        dataset = CustomDataset(
            ids=mock_ids,
            image_tensors=mock_image_tensors,
            labels=mock_labels
        )
        
        # 첫 번째 아이템 접근
        sample_id, image_tensor, label_tensor = dataset[0]
        
        assert sample_id == 'sample1'
        assert image_tensor.shape == (10, 3, 256, 256)
        assert label_tensor.shape == (3,)
    
    def test_get_sample_by_id(self):
        """ID로 샘플 반환 테스트"""
        mock_ids = ['sample1', 'sample2']
        mock_image_tensors = torch.randn(2, 10, 3, 256, 256)
        mock_labels = torch.tensor([[1, 0, 0], [0, 1, 0]])
        
        dataset = CustomDataset(
            ids=mock_ids,
            image_tensors=mock_image_tensors,
            labels=mock_labels
        )
        
        # ID로 샘플 찾기
        image_tensor, label_tensor = dataset.get_sample_by_id('sample1')
        
        assert image_tensor.shape == (10, 3, 256, 256)
        assert label_tensor.shape == (3,)
    
    def test_get_class_distribution(self):
        """클래스 분포 반환 테스트"""
        mock_ids = ['sample1', 'sample2', 'sample3']
        mock_image_tensors = torch.randn(3, 10, 3, 256, 256)
        mock_labels = torch.tensor([[1, 0, 0], [0, 1, 0], [1, 0, 0]])
        
        dataset = CustomDataset(
            ids=mock_ids,
            image_tensors=mock_image_tensors,
            labels=mock_labels
        )
        
        # 클래스 분포 계산
        class_dist = dataset.get_class_distribution()
        
        assert class_dist[0] == 2  # 클래스 0: 2개
        assert class_dist[1] == 1  # 클래스 1: 1개
    
    def test_invalid_sample_id(self):
        """존재하지 않는 ID로 샘플 접근 테스트"""
        mock_ids = ['sample1']
        mock_image_tensors = torch.randn(1, 10, 3, 256, 256)
        mock_labels = torch.tensor([[1, 0, 0]])
        
        dataset = CustomDataset(
            ids=mock_ids,
            image_tensors=mock_image_tensors,
            labels=mock_labels
        )
        
        # 존재하지 않는 ID로 접근
        with pytest.raises(ValueError):
            dataset.get_sample_by_id('nonexistent_id')


# 실제 존재하는 클래스들만 테스트하므로 추가 테스트 클래스는 제거 