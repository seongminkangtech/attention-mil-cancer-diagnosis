"""
데이터셋 모듈

의료 영상 데이터를 처리하는 CustomDataset 클래스를 정의합니다.
"""

from typing import List, Tuple

import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """
    의료 영상 데이터셋

    의료 영상 프레임과 라벨을 처리하는 커스텀 데이터셋 클래스입니다.
    """

    def __init__(
        self, ids: List[str], image_tensors: torch.Tensor, labels: torch.Tensor
    ):
        """
        데이터셋 초기화

        Args:
            ids (List[str]): 샘플 ID 리스트
            image_tensors (torch.Tensor): 이미지 텐서 (N, num_frames, C, H, W)
            labels (torch.Tensor): 라벨 텐서 (N, num_classes)
        """
        self.ids = ids
        self.image_tensors = image_tensors
        self.labels = labels

    def __len__(self) -> int:
        """
        데이터셋 크기 반환

        Returns:
            int: 데이터셋 크기
        """
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor, torch.Tensor]:
        """
        샘플 반환

        Args:
            idx (int): 샘플 인덱스

        Returns:
            Tuple[str, torch.Tensor, torch.Tensor]: (ID, 이미지 텐서, 라벨 텐서)
        """
        sample_id = self.ids[idx]
        image_tensor = self.image_tensors[idx]
        label_tensor = self.labels[idx]

        return sample_id, image_tensor, label_tensor

    def get_sample_by_id(self, sample_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ID로 샘플 반환

        Args:
            sample_id (str): 샘플 ID

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (이미지 텐서, 라벨 텐서)
        """
        if sample_id in self.ids:
            idx = self.ids.index(sample_id)
            return self.image_tensors[idx], self.labels[idx]
        else:
            raise ValueError(f"Sample ID '{sample_id}' not found in dataset")

    def get_class_distribution(self) -> dict:
        """
        클래스 분포 반환

        Returns:
            dict: 클래스별 샘플 수
        """
        class_counts = {}
        for label in self.labels:
            class_idx = torch.argmax(label).item()
            class_counts[class_idx] = class_counts.get(class_idx, 0) + 1

        return class_counts
