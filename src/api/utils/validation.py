"""
입력 검증 유틸리티

설정 파일 및 입력 데이터 검증을 위한 함수들을 포함합니다.
"""

from typing import Any, Dict, List, Optional
import os
import yaml
from fastapi import HTTPException


def validate_config(config: Dict[str, Any]) -> bool:
    """
    설정 파일 검증
    
    Args:
        config (Dict[str, Any]): 검증할 설정 딕셔너리
        
    Returns:
        bool: 검증 성공 여부
        
    Raises:
        ValueError: 설정이 유효하지 않은 경우
    """
    required_keys = [
        'model',
        'data',
        'training',
        'paths',
        'hardware'
    ]
    
    # 필수 키 확인
    for key in required_keys:
        if key not in config:
            raise ValueError(f"필수 설정 키가 누락되었습니다: {key}")
    
    # 모델 설정 검증
    model_config = config.get('model', {})
    if 'num_classes' not in model_config:
        raise ValueError("모델 설정에 num_classes가 필요합니다.")
    
    # 데이터 설정 검증
    data_config = config.get('data', {})
    if 'image_count' not in data_config:
        raise ValueError("데이터 설정에 image_count가 필요합니다.")
    
    # 경로 설정 검증
    paths_config = config.get('paths', {})
    required_paths = ['train_csv', 'test_csv', 'label_csv', 'frame_path']
    for path_key in required_paths:
        if path_key not in paths_config:
            raise ValueError(f"경로 설정에 {path_key}가 필요합니다.")
    
    return True


def validate_image_input(
    image_files: List[str],
    max_files: int = 50,
    allowed_extensions: List[str] = None
) -> bool:
    """
    이미지 입력 검증
    
    Args:
        image_files (List[str]): 이미지 파일 경로 리스트
        max_files (int): 최대 파일 수
        allowed_extensions (List[str]): 허용된 파일 확장자
        
    Returns:
        bool: 검증 성공 여부
        
    Raises:
        ValueError: 입력이 유효하지 않은 경우
    """
    if allowed_extensions is None:
        allowed_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    # 파일 수 검증
    if len(image_files) > max_files:
        raise ValueError(f"이미지 파일 수가 최대 허용치를 초과했습니다. (최대: {max_files})")
    
    if len(image_files) == 0:
        raise ValueError("최소 하나의 이미지 파일이 필요합니다.")
    
    # 파일 존재 여부 및 확장자 검증
    for file_path in image_files:
        if not os.path.exists(file_path):
            raise ValueError(f"파일이 존재하지 않습니다: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in allowed_extensions:
            raise ValueError(f"지원하지 않는 파일 형식입니다: {file_ext}")
    
    return True


def validate_prediction_request(
    request_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    추론 요청 데이터 검증
    
    Args:
        request_data (Dict[str, Any]): 검증할 요청 데이터
        
    Returns:
        Dict[str, Any]: 검증된 데이터
        
    Raises:
        HTTPException: 검증 실패 시
    """
    # 필수 필드 확인
    required_fields = ['image_paths']
    for field in required_fields:
        if field not in request_data:
            raise HTTPException(
                status_code=400,
                detail=f"필수 필드가 누락되었습니다: {field}"
            )
    
    # 이미지 경로 검증
    image_paths = request_data.get('image_paths', [])
    if not isinstance(image_paths, list):
        raise HTTPException(
            status_code=400,
            detail="image_paths는 리스트 형태여야 합니다."
        )
    
    try:
        validate_image_input(image_paths)
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    
    # 옵션 필드 설정
    validated_data = {
        'image_paths': image_paths,
        'confidence_threshold': request_data.get('confidence_threshold', 0.5),
        'return_attention': request_data.get('return_attention', False),
        'batch_size': request_data.get('batch_size', 1)
    }
    
    return validated_data


def validate_model_config(config_path: str) -> Dict[str, Any]:
    """
    모델 설정 파일 검증 및 로드
    
    Args:
        config_path (str): 설정 파일 경로
        
    Returns:
        Dict[str, Any]: 검증된 설정 딕셔너리
        
    Raises:
        HTTPException: 설정 파일 로드 실패 시
    """
    try:
        if not os.path.exists(config_path):
            raise HTTPException(
                status_code=500,
                detail=f"설정 파일을 찾을 수 없습니다: {config_path}"
            )
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        validate_config(config)
        return config
        
    except yaml.YAMLError as e:
        raise HTTPException(
            status_code=500,
            detail=f"설정 파일 파싱 오류: {str(e)}"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=500,
            detail=f"설정 파일 검증 오류: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"설정 파일 로드 오류: {str(e)}"
        ) 