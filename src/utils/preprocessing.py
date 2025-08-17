"""
전처리 모듈

의료 영상 데이터의 전처리 함수들을 정의합니다.
"""

import os
from glob import glob
from typing import List, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor
from tqdm import tqdm


def load_and_preprocess_images(
    data_path: str,
    csv_file: str,
    label_file: str,
    image_count: int = 50,
    img_size: int = 256,
) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
    """
    의료 영상 데이터 로드 및 전처리

    Args:
        data_path (str): 데이터 경로
        csv_file (str): CSV 파일 경로
        label_file (str): 라벨 파일 경로
        image_count (int): 샘플링할 프레임 수 (기본값: 50)
        img_size (int): 이미지 크기 (기본값: 256)

    Returns:
        Tuple[List[str], torch.Tensor, torch.Tensor]: (ID 리스트, 이미지 텐서, 라벨 텐서)
    """
    # 데이터 로드
    data = pd.read_csv(csv_file, encoding="cp949")
    label_data = pd.read_csv(label_file, encoding="cp949")

    # 이미지 경로 생성
    image_list = []
    for i in range(len(data)):
        file_name = data.loc[i]["FileName"]
        id_name = file_name[: file_name.find("_")]
        image_list.append(os.path.join(data_path, id_name))

    # 라벨 및 ID 리스트 생성
    label_list = []
    id_list = []
    image_tensor = torch.empty((len(image_list), image_count, 3, img_size, img_size))

    # 변환 함수
    tf = ToTensor()

    print(f"데이터 로딩 중... (총 {len(image_list)}개 샘플)")

    # 각 샘플 처리
    for i in tqdm(range(len(image_list)), desc="이미지 처리"):
        folder_name = os.path.basename(image_list[i])

        # 라벨 정보 추출
        dst_label = label_data.loc[label_data["일련번호"] == int(folder_name[:-1])]
        dst_label = dst_label.loc[
            dst_label["구분값"] == int(folder_name[-1])
        ].reset_index()
        label = int(dst_label.loc[0]["OTE 원인"])

        id_list.append(folder_name)
        label_list.append(label - 1)  # 0-based 인덱스로 변환

        # 이미지 파일 리스트
        image_file_list = glob(os.path.join(str(image_list[i]), "*.jpg"))

        if len(image_file_list) > image_count:
            # 랜덤 샘플링
            image_index = torch.randint(
                low=0, high=len(image_file_list) - image_count, size=(1,)
            )
            start_idx = image_index.item()

            for count in range(image_count):
                image = Image.open(image_file_list[int(start_idx + count)]).resize(
                    (img_size, img_size)
                )
                image = 1 - tf(image)  # 색상 반전
                image_tensor[i, count] = image
        else:
            # 모든 이미지 사용 후 부족한 부분 반복
            for count in range(len(image_file_list)):
                image = Image.open(image_file_list[count]).resize((img_size, img_size))
                image = 1 - tf(image)
                image_tensor[i, count] = image

            # 부족한 프레임 반복으로 채우기
            for j in range(image_count - len(image_file_list)):
                image = Image.open(image_file_list[j % len(image_file_list)]).resize(
                    (img_size, img_size)
                )
                image = 1 - tf(image)
                image_tensor[i, int(len(image_file_list) + j)] = image

    # 라벨을 one-hot 인코딩으로 변환
    labels = F.one_hot(torch.tensor(label_list).to(torch.int64), num_classes=3)

    return id_list, image_tensor, labels


def create_data_loaders(
    id_list: List[str],
    image_tensor: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int = 4,
    train_ratio: float = 0.8,
    shuffle: bool = True,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    데이터 로더 생성

    Args:
        id_list (List[str]): 샘플 ID 리스트
        image_tensor (torch.Tensor): 이미지 텐서
        labels (torch.Tensor): 라벨 텐서
        batch_size (int): 배치 크기 (기본값: 4)
        train_ratio (float): 훈련 데이터 비율 (기본값: 0.8)
        shuffle (bool): 셔플 여부 (기본값: True)

    Returns:
        Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]: (훈련 로더, 검증 로더)
    """
    from .dataset import CustomDataset

    # 데이터셋 생성
    dataset = CustomDataset(id_list, image_tensor, labels)

    # 훈련/검증 분할
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # 데이터 로더 생성
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )

    return train_loader, val_loader


def get_class_names() -> List[str]:
    """
    클래스 이름 반환

    Returns:
        List[str]: 클래스 이름 리스트
    """
    return ["Oropharynx", "Tonguebase", "Epiglottis"]


def normalize_image(image: torch.Tensor) -> torch.Tensor:
    """
    이미지 정규화

    Args:
        image (torch.Tensor): 입력 이미지 텐서

    Returns:
        torch.Tensor: 정규화된 이미지 텐서
    """
    # 이미지가 [0, 1] 범위에 있다고 가정
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    return (image - mean) / std
