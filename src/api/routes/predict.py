"""
추론 엔드포인트

의료 영상 분류를 위한 추론 API입니다.
"""

import os
import sys
import numpy as np
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
import time
from datetime import datetime
from PIL import Image
import logging

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.models.onnx_model import ONNXModel
from src.utils import get_class_names
from ..utils.response import create_response, create_error_response
from ..utils.validation import validate_prediction_request, validate_model_config

logger = logging.getLogger(__name__)


# Pydantic 모델 정의
class PredictionRequest(BaseModel):
    """추론 요청 모델"""
    image_paths: List[str]
    confidence_threshold: Optional[float] = 0.5
    return_attention: Optional[bool] = False
    batch_size: Optional[int] = 1


class PredictionResponse(BaseModel):
    """추론 응답 모델"""
    success: bool
    message: str
    timestamp: str
    service: str
    data: Optional[Dict[str, Any]] = None
    error_code: Optional[str] = None


router = APIRouter()

# 전역 변수로 모델과 설정 저장
_model = None
_config = None
_class_names = None


def load_onnx_model(config: Dict[str, Any]) -> ONNXModel:
    """
    설정을 기반으로 ONNX 모델 로드
    
    Args:
        config (Dict[str, Any]): 모델 설정
        
    Returns:
        ONNXModel: 로드된 ONNX 모델
    """
    try:
        # ONNX 모델 경로 설정
        onnx_model_path = config.get('deployment', {}).get('onnx_model_path', 'models/attention_mil.onnx')
        
        # ONNX 모델 로드
        model = ONNXModel(onnx_model_path)
        logger.info(f"✅ ONNX 모델 로드 성공: {onnx_model_path}")
        
        return model
        
    except Exception as e:
        logger.error(f"❌ ONNX 모델 로드 실패: {e}")
        raise


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
        
        # ONNX 모델 로드
        _model = load_onnx_model(_config)
        logger.info("✅ ONNX 모델을 성공적으로 로드했습니다.")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 모델 로드 실패: {e}")
        return False


def preprocess_images_for_prediction(
    image_paths: List[str],
    config: Dict[str, Any]
) -> np.ndarray:
    """
    추론을 위한 이미지 전처리
    
    Args:
        image_paths (List[str]): 이미지 파일 경로 리스트
        config (Dict[str, Any]): 설정 딕셔너리
        
    Returns:
        np.ndarray: 전처리된 이미지 배열
    """
    # 설정에서 파라미터 추출
    image_count = config['data']['image_count']
    img_size = config['data']['img_size']
    
    # 이미지 로드 및 전처리
    processed_images = []
    for img_path in image_paths[:image_count]:  # 최대 image_count개만 사용
        try:
            image = Image.open(img_path).convert('RGB')
            image = image.resize((img_size, img_size))
            
            # PIL 이미지를 numpy 배열로 변환
            image_array = np.array(image, dtype=np.float32) / 255.0
            
            # 색상 반전 (기존 전처리와 동일)
            image_array = 1 - image_array
            
            # (H, W, C) -> (C, H, W) 변환
            image_array = image_array.transpose(2, 0, 1)
            
            processed_images.append(image_array)
            
        except Exception as e:
            logger.error(f"이미지 처리 실패: {img_path}, 오류: {e}")
            continue
    
    # 부족한 이미지는 첫 번째 이미지로 채우기
    while len(processed_images) < image_count:
        if processed_images:
            processed_images.append(processed_images[0])
        else:
            # 빈 이미지 생성
            empty_image = np.zeros((3, img_size, img_size), dtype=np.float32)
            processed_images.append(empty_image)
    
    # 배치 차원 추가
    image_array = np.stack(processed_images[:image_count])
    image_array = np.expand_dims(image_array, axis=0)  # (1, image_count, 3, img_size, img_size)
    
    return image_array


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
        image_array = preprocess_images_for_prediction(image_paths, _config)
        
        # ONNX 모델로 추론 수행
        prediction, attention_weights = _model.predict(image_array)
        
        # 출력 후처리
        postprocessed_result = _model.postprocess_output(prediction, _class_names)
        
        # 예측 결과
        predicted_class = postprocessed_result['predicted_class']
        confidence = postprocessed_result['confidence']
        class_probabilities = postprocessed_result['class_probabilities']
        
        # Attention 가중치 (요청된 경우)
        if return_attention and attention_weights is not None:
            attention_weights = attention_weights[0].tolist()
        
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
                "model_name": "Attention MIL (ONNX)",
                "num_classes": len(_class_names),
                "model_type": "ONNX"
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
        "model_name": "Attention MIL (ONNX)",
        "num_classes": len(_class_names) if _class_names else 0,
        "class_names": _class_names,
        "model_loaded": _model is not None,
        "config_loaded": _config is not None
    }
    
    if _model:
        model_info.update(_model.get_model_info())
    
    if _config:
        model_info.update({
            "image_count": _config['data']['image_count'],
            "image_size": _config['data']['img_size']
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
                    "model_type": "ONNX"
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