"""
ONNX 모델 로더 및 추론기

ONNX 형식으로 변환된 모델을 로드하고 추론을 수행합니다.
"""

import os
import numpy as np
import onnxruntime as ort
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ONNXModel:
    """
    ONNX 모델을 로드하고 추론을 수행하는 클래스
    """
    
    def __init__(self, model_path: str, providers: Optional[List[str]] = None):
        """
        ONNX 모델 초기화
        
        Args:
            model_path (str): ONNX 모델 파일 경로
            providers (Optional[List[str]]): ONNX Runtime 실행 제공자 목록
        """
        self.model_path = model_path
        self.session = None
        self.input_names = []
        self.output_names = []
        self.input_shapes = []
        self.output_shapes = []
        
        # 기본 제공자 설정 (GPU 우선, CPU 폴백)
        if providers is None:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        self.providers = providers
        self.load_model()
    
    def load_model(self):
        """ONNX 모델 로드"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"ONNX 모델 파일을 찾을 수 없습니다: {self.model_path}")
            
            # ONNX Runtime 세션 생성
            self.session = ort.InferenceSession(
                self.model_path, 
                providers=self.providers
            )
            
            # 입력/출력 정보 추출
            self.input_names = [input.name for input in self.session.get_inputs()]
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            # 입력/출력 형태 정보
            for input_info in self.session.get_inputs():
                self.input_shapes.append(input_info.shape)
            
            for output_info in self.session.get_outputs():
                self.output_shapes.append(output_info.shape)
            
            logger.info(f"✅ ONNX 모델 로드 성공: {self.model_path}")
            logger.info(f"   - 입력: {self.input_names}")
            logger.info(f"   - 출력: {self.output_names}")
            logger.info(f"   - 제공자: {self.session.get_providers()}")
            
        except Exception as e:
            logger.error(f"❌ ONNX 모델 로드 실패: {e}")
            raise
    
    def predict(self, input_data: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        추론 수행
        
        Args:
            input_data (np.ndarray): 입력 데이터
            
        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]: (예측 결과, attention 가중치)
        """
        try:
            if self.session is None:
                raise RuntimeError("모델이 로드되지 않았습니다.")
            
            # 입력 데이터 타입 검증
            if not isinstance(input_data, np.ndarray):
                raise ValueError(f"입력 데이터는 numpy 배열이어야 합니다. 현재 타입: {type(input_data)}")
            
            # 입력 데이터 형태 검증
            if input_data.ndim != 5:
                raise ValueError(f"입력 데이터는 5차원이어야 합니다. 현재 차원: {input_data.ndim}")
            
            # 입력 데이터 준비
            input_feed = {self.input_names[0]: input_data}
            
            # 추론 실행
            outputs = self.session.run(self.output_names, input_feed)
            
            # 출력 처리
            if len(outputs) == 1:
                # 단일 출력 (예측 결과만)
                prediction = outputs[0]
                attention_weights = None
            else:
                # 다중 출력 (예측 결과 + attention 가중치)
                prediction = outputs[0]
                attention_weights = outputs[1] if len(outputs) > 1 else None
            
            return prediction, attention_weights
            
        except Exception as e:
            logger.error(f"❌ 추론 실패: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        모델 정보 반환
        
        Returns:
            Dict[str, Any]: 모델 정보
        """
        return {
            "model_type": "ONNX",
            "model_path": self.model_path,
            "input_names": self.input_names,
            "output_names": self.output_names,
            "input_shapes": self.input_shapes,
            "output_shapes": self.output_shapes,
            "providers": self.session.get_providers() if self.session else [],
            "loaded": self.session is not None
        }
    
    def preprocess_input(self, images: List[np.ndarray], target_shape: Tuple[int, ...]) -> np.ndarray:
        """
        입력 이미지 전처리
        
        Args:
            images (List[np.ndarray]): 입력 이미지 리스트
            target_shape (Tuple[int, ...]): 목표 형태
            
        Returns:
            np.ndarray: 전처리된 입력 데이터
        """
        try:
            # 이미지 개수 확인
            if len(images) == 0:
                raise ValueError("입력 이미지가 없습니다.")
            
            # 첫 번째 이미지의 형태로 모든 이미지 통일
            processed_images = []
            for img in images:
                if img.ndim == 3:  # (H, W, C)
                    img = img.transpose(2, 0, 1)  # (C, H, W)
                processed_images.append(img)
            
            # 배치 차원 추가 및 스택
            if len(target_shape) == 4:  # (batch, tiles, channels, height, width)
                batch_data = np.stack(processed_images, axis=0)
                batch_data = np.expand_dims(batch_data, axis=0)  # (1, tiles, C, H, W)
            else:
                batch_data = np.stack(processed_images, axis=0)
            
            return batch_data.astype(np.float32)
            
        except Exception as e:
            logger.error(f"❌ 입력 전처리 실패: {e}")
            raise
    
    def postprocess_output(self, prediction: np.ndarray, class_names: List[str]) -> Dict[str, Any]:
        """
        출력 후처리
        
        Args:
            prediction (np.ndarray): 모델 출력
            class_names (List[str]): 클래스 이름 리스트
            
        Returns:
            Dict[str, Any]: 후처리된 결과
        """
        try:
            # 소프트맥스 적용
            exp_pred = np.exp(prediction - np.max(prediction, axis=1, keepdims=True))
            probabilities = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)
            
            # 예측 클래스 및 신뢰도
            predicted_class = np.argmax(probabilities, axis=1)
            confidence = np.max(probabilities, axis=1)
            
            # 클래스별 확률
            class_probabilities = {}
            for i, class_name in enumerate(class_names):
                if i < probabilities.shape[1]:
                    class_probabilities[class_name] = float(probabilities[0, i])
            
            return {
                "predicted_class": int(predicted_class[0]),
                "confidence": float(confidence[0]),
                "class_probabilities": class_probabilities,
                "raw_probabilities": probabilities[0].tolist()
            }
            
        except Exception as e:
            logger.error(f"❌ 출력 후처리 실패: {e}")
            raise 