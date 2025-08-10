"""
추론 엔드포인트

의료 영상 분류를 위한 추론 API입니다.
"""

import os
import sys
import torch
import numpy as np
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
import mlflow.pytorch
import time
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.models import AttentionMILModel, FeatureExtractor
from src.utils import load_and_preprocess_images, get_class_names
from ..utils.response import create_response, create_error_response
from ..utils.validation import validate_prediction_request, validate_model_config


# Pydantic 모델 정의
class PredictionRequest(BaseModel):
    """추론 요청 모델"""
    image_paths: List[str]
    confidence_threshold: Optional[float] = 0.5
    return_attention: Optional[bool] = False
    batch_size: Optional[int] = 1


class PredictionResponse(BaseModel):
    """추론 응답 모델"""
    prediction: str
    confidence: float
    class_probabilities: Dict[str, float]
    attention_weights: Optional[List[float]] = None
    processing_time: float


router = APIRouter()

# 전역 변수로 모델과 설정 저장
_model = None
_config = None
_class_names = None


def load_model_from_mlflow():
    """
    MLflow에서 모델 로드
    
    Returns:
        AttentionMILModel: 로드된 모델
    """
    try:
        # MLflow에서 최신 모델 로드
        # 실제 구현에서는 MLflow 모델 레지스트리에서 로드
        model_uri = "runs:/latest/model"
        model = mlflow.pytorch.load_model(model_uri)
        return model
    except Exception as e:
        print(f"MLflow에서 모델 로드 실패: {e}")
        return None


def create_model_from_config(config: Dict[str, Any]) -> AttentionMILModel:
    """
    설정을 기반으로 모델 생성
    
    Args:
        config (Dict[str, Any]): 모델 설정
        
    Returns:
        AttentionMILModel: 생성된 모델
    """
    # 특징 추출기 생성
    feature_extractor = FeatureExtractor(
        model_name=config['model']['feature_extractor']['model_name'],
        pretrained=config['model']['feature_extractor']['pretrained']
    )
    
    # Attention MIL 모델 생성
    model = AttentionMILModel(
        num_classes=config['model']['num_classes'],
        feature_extractor=feature_extractor,
        attention_hidden_dim=config['model']['attention']['hidden_dim'],
        dropout_rate=config['model']['attention']['dropout_rate']
    )
    
    return model


def load_model():
    """
    모델 로드 (전역 변수에 저장)
    """
    global _model, _config, _class_names
    
    try:
        # 설정 파일 로드
        config_path = "configs/model_configs/attention_mil.yaml"
        _config = validate_model_config(config_path)
        
        # 클래스 이름 로드
        _class_names = get_class_names()
        
        # MLflow에서 모델 로드 시도
        _model = load_model_from_mlflow()
        
        if _model is None:
            # MLflow 로드 실패 시 설정으로 모델 생성
            _model = create_model_from_config(_config)
            print("⚠️ MLflow에서 모델을 로드할 수 없어 설정으로 모델을 생성했습니다.")
        else:
            print("✅ MLflow에서 모델을 성공적으로 로드했습니다.")
        
        # 모델을 평가 모드로 설정
        _model.eval()
        
        # GPU 사용 가능 시 GPU로 이동
        if torch.cuda.is_available():
            _model = _model.cuda()
            print("✅ 모델을 GPU로 이동했습니다.")
        
        return True
        
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return False


def preprocess_images_for_prediction(
    image_paths: List[str],
    config: Dict[str, Any]
) -> torch.Tensor:
    """
    추론을 위한 이미지 전처리
    
    Args:
        image_paths (List[str]): 이미지 파일 경로 리스트
        config (Dict[str, Any]): 설정 딕셔너리
        
    Returns:
        torch.Tensor: 전처리된 이미지 텐서
    """
    # 설정에서 파라미터 추출
    image_count = config['data']['image_count']
    img_size = config['data']['img_size']
    
    # 이미지 전처리 (실제 구현에서는 더 정교한 전처리 필요)
    from PIL import Image
    from torchvision.transforms import ToTensor, Resize
    
    transform = ToTensor()
    resize = Resize((img_size, img_size))
    
    # 이미지 로드 및 전처리
    processed_images = []
    for img_path in image_paths[:image_count]:  # 최대 image_count개만 사용
        try:
            image = Image.open(img_path).convert('RGB')
            image = resize(image)
            image = transform(image)
            image = 1 - image  # 색상 반전 (기존 전처리와 동일)
            processed_images.append(image)
        except Exception as e:
            print(f"이미지 처리 실패: {img_path}, 오류: {e}")
            continue
    
    # 부족한 이미지는 첫 번째 이미지로 채우기
    while len(processed_images) < image_count:
        if processed_images:
            processed_images.append(processed_images[0])
        else:
            # 빈 이미지 생성
            empty_image = torch.zeros(3, img_size, img_size)
            processed_images.append(empty_image)
    
    # 배치 차원 추가
    image_tensor = torch.stack(processed_images[:image_count])
    image_tensor = image_tensor.unsqueeze(0)  # (1, image_count, 3, img_size, img_size)
    
    return image_tensor


@router.post("/", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    의료 영상 분류 추론
    
    Args:
        request (PredictionRequest): 추론 요청 데이터
        background_tasks (BackgroundTasks): 백그라운드 작업
        
    Returns:
        Dict[str, Any]: 추론 결과
    """
    global _model, _config, _class_names
    
    start_time = time.time()
    
    try:
        # 모델이 로드되지 않은 경우 로드
        if _model is None:
            if not load_model():
                return create_error_response(
                    message="모델을 로드할 수 없습니다.",
                    error_code="MODEL_LOAD_ERROR"
                )
        
        # 입력 데이터 검증
        validated_data = validate_prediction_request(request.dict())
        image_paths = validated_data['image_paths']
        confidence_threshold = validated_data['confidence_threshold']
        return_attention = validated_data['return_attention']
        
        # 이미지 전처리
        image_tensor = preprocess_images_for_prediction(image_paths, _config)
        
        # GPU 사용 가능 시 GPU로 이동
        if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()
        
        # 추론 수행
        with torch.no_grad():
            outputs = _model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            # 예측 결과
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
            
            # 클래스별 확률
            class_probabilities = {}
            for i, prob in enumerate(probabilities[0]):
                class_probabilities[_class_names[i]] = prob.item()
            
            # Attention 가중치 (요청된 경우)
            attention_weights = None
            if return_attention:
                attention_weights = _model.get_attention_weights(image_tensor)
                attention_weights = attention_weights[0].cpu().numpy().tolist()
            
            # 신뢰도 임계값 확인
            if confidence < confidence_threshold:
                prediction_result = "uncertain"
            else:
                prediction_result = _class_names[predicted_class]
        
        processing_time = time.time() - start_time
        
        # 응답 생성
        response_data = {
            "prediction": prediction_result,
            "confidence": confidence,
            "class_probabilities": class_probabilities,
            "processing_time": processing_time,
            "input_image_count": len(image_paths),
            "model_info": {
                "model_name": "Attention MIL",
                "num_classes": len(_class_names),
                "device": "cuda" if torch.cuda.is_available() else "cpu"
            }
        }
        
        if attention_weights:
            response_data["attention_weights"] = attention_weights
        
        return create_response(
            success=True,
            message="추론이 성공적으로 완료되었습니다.",
            data=response_data
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        return create_error_response(
            message=f"추론 중 오류가 발생했습니다: {str(e)}",
            error_code="PREDICTION_ERROR",
            details={
                "processing_time": processing_time,
                "error_type": type(e).__name__
            }
        )


@router.get("/model-info")
async def get_model_info() -> Dict[str, Any]:
    """
    모델 정보 조회
    
    Returns:
        Dict[str, Any]: 모델 정보
    """
    global _model, _config, _class_names
    
    if _model is None:
        return create_error_response(
            message="모델이 로드되지 않았습니다.",
            error_code="MODEL_NOT_LOADED"
        )
    
    model_info = {
        "model_name": "Attention MIL",
        "num_classes": len(_class_names) if _class_names else 0,
        "class_names": _class_names,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "model_loaded": _model is not None,
        "config_loaded": _config is not None
    }
    
    if _config:
        model_info.update({
            "image_count": _config['data']['image_count'],
            "image_size": _config['data']['img_size'],
            "feature_extractor": _config['model']['feature_extractor']['model_name']
        })
    
    return create_response(
        success=True,
        message="모델 정보 조회 완료",
        data=model_info
    )


@router.post("/reload-model")
async def reload_model() -> Dict[str, Any]:
    """
    모델 재로드
    
    Returns:
        Dict[str, Any]: 재로드 결과
    """
    global _model, _config, _class_names
    
    try:
        # 기존 모델 정리
        if _model is not None:
            del _model
            _model = None
        
        # 모델 재로드
        success = load_model()
        
        if success:
            return create_response(
                success=True,
                message="모델이 성공적으로 재로드되었습니다.",
                data={
                    "model_loaded": True,
                    "device": "cuda" if torch.cuda.is_available() else "cpu"
                }
            )
        else:
            return create_error_response(
                message="모델 재로드에 실패했습니다.",
                error_code="MODEL_RELOAD_ERROR"
            )
            
    except Exception as e:
        return create_error_response(
            message=f"모델 재로드 중 오류가 발생했습니다: {str(e)}",
            error_code="MODEL_RELOAD_ERROR"
        ) 