#!/usr/bin/env python3
"""
ONNX 도구 통합 스크립트

모델 변환, 최적화, 검증 등 ONNX 관련 모든 작업을 수행합니다.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import onnx
import onnxruntime as ort
import torch

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from configs import AppConfig
from src.models.attention_mil import AttentionMIL

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ONNXConverter:
    """PyTorch 모델을 ONNX로 변환하는 클래스"""

    def __init__(self, config: AppConfig):
        self.config = config
        self.model = None
        self.device = torch.device(config.attention_mil.hardware.device)

    def load_model(self, model_path: str) -> bool:
        """학습된 모델 로드"""
        try:
            # 모델 아키텍처 생성
            self.model = AttentionMIL(
                num_classes=self.config.attention_mil.num_classes,
                feature_extractor_config=self.config.attention_mil.feature_extractor,
                attention_config=self.config.attention_mil.attention,
            )

            # 가중치 로드
            checkpoint = torch.load(model_path, map_location=self.device)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)

            self.model.to(self.device)
            self.model.eval()
            logger.info(f"모델 로드 완료: {model_path}")
            return True

        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
            return False

    def convert_to_onnx(
        self, output_path: str, input_shape: tuple = (1, 3, 224, 224)
    ) -> bool:
        """PyTorch 모델을 ONNX로 변환"""
        try:
            if self.model is None:
                logger.error("모델이 로드되지 않았습니다.")
                return False

            # 더미 입력 생성
            dummy_input = torch.randn(input_shape, device=self.device)

            # ONNX 변환
            torch.onnx.export(
                self.model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            )

            logger.info(f"ONNX 변환 완료: {output_path}")
            return True

        except Exception as e:
            logger.error(f"ONNX 변환 실패: {e}")
            return False


class ONNXOptimizer:
    """ONNX 모델 최적화 클래스"""

    def __init__(self, config: AppConfig):
        self.config = config

    def optimize_model(self, input_path: str, output_path: str) -> bool:
        """ONNX 모델 최적화"""
        try:
            # ONNX 모델 로드
            model = onnx.load(input_path)

            # 기본 최적화
            from onnxsim import simplify

            model_sim, check = simplify(model)

            if check:
                onnx.save(model_sim, output_path)
                logger.info(f"모델 최적화 완료: {output_path}")
                return True
            else:
                logger.warning("모델 최적화 검증 실패, 원본 모델 저장")
                onnx.save(model, output_path)
                return True

        except Exception as e:
            logger.error(f"모델 최적화 실패: {e}")
            return False

    def quantize_model(self, input_path: str, output_path: str) -> bool:
        """ONNX 모델 양자화 (INT8)"""
        try:
            # ONNX Runtime 양자화
            from onnxruntime.quantization import quantize_dynamic

            quantize_dynamic(
                model_input=input_path,
                model_output=output_path,
                weight_type=onnx.TensorProto.INT8,
                optimize_model=True,
            )

            logger.info(f"모델 양자화 완료: {output_path}")
            return True

        except Exception as e:
            logger.error(f"모델 양자화 실패: {e}")
            return False


class ONNXValidator:
    """ONNX 모델 검증 클래스"""

    def __init__(self, config: AppConfig):
        self.config = config

    def validate_model(self, model_path: str) -> Dict[str, Any]:
        """ONNX 모델 유효성 검증"""
        try:
            # ONNX 모델 로드 및 검증
            model = onnx.load(model_path)
            onnx.checker.check_model(model)

            # 모델 정보 수집
            model_info = {
                "ir_version": model.ir_version,
                "opset_version": model.opset_version,
                "producer_name": model.producer_name,
                "input_count": len(model.graph.input),
                "output_count": len(model.graph.output),
                "node_count": len(model.graph.node),
            }

            # 입력/출력 정보
            for i, input_info in enumerate(model.graph.input):
                model_info[f"input_{i}_shape"] = str(input_info.type.tensor_type.shape)
                model_info[f"input_{i}_type"] = str(
                    input_info.type.tensor_type.elem_type
                )

            for i, output_info in enumerate(model.graph.output):
                model_info[f"output_{i}_shape"] = str(
                    output_info.type.tensor_type.shape
                )
                model_info[f"output_{i}_type"] = str(
                    output_info.type.tensor_type.elem_type
                )

            logger.info("ONNX 모델 검증 완료")
            return model_info

        except Exception as e:
            logger.error(f"ONNX 모델 검증 실패: {e}")
            return {}

    def test_inference(
        self, model_path: str, input_shape: tuple = (1, 3, 224, 224)
    ) -> bool:
        """ONNX 모델 추론 테스트"""
        try:
            # ONNX Runtime 세션 생성
            session = ort.InferenceSession(model_path)

            # 더미 입력 생성
            import numpy as np

            dummy_input = np.random.randn(*input_shape).astype(np.float32)

            # 추론 실행
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name

            result = session.run([output_name], {input_name: dummy_input})

            logger.info(f"추론 테스트 성공: 출력 형태 {result[0].shape}")
            return True

        except Exception as e:
            logger.error(f"추론 테스트 실패: {e}")
            return False


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="ONNX 도구 통합 스크립트")
    parser.add_argument(
        "--action",
        required=True,
        choices=["convert", "optimize", "quantize", "validate", "test"],
        help="수행할 작업",
    )
    parser.add_argument("--model_path", required=True, help="PyTorch 모델 경로")
    parser.add_argument("--output_path", help="출력 파일 경로")
    parser.add_argument(
        "--input_shape", default="1,3,224,224", help="입력 형태 (쉼표로 구분)"
    )
    parser.add_argument(
        "--config", default="configs/attention_mil.yaml", help="설정 파일 경로"
    )

    args = parser.parse_args()

    # 설정 로드
    try:
        config = AppConfig.from_yaml(args.config)
    except Exception as e:
        logger.error(f"설정 로드 실패: {e}")
        return

    # 입력 형태 파싱
    input_shape = tuple(map(int, args.input_shape.split(",")))

    if args.action == "convert":
        # PyTorch -> ONNX 변환
        converter = ONNXConverter(config)
        if converter.load_model(args.model_path):
            output_path = args.output_path or args.model_path.replace(".pth", ".onnx")
            converter.convert_to_onnx(output_path, input_shape)

    elif args.action == "optimize":
        # ONNX 모델 최적화
        optimizer = ONNXOptimizer(config)
        output_path = args.output_path or args.model_path.replace(
            ".onnx", "_optimized.onnx"
        )
        optimizer.optimize_model(args.model_path, output_path)

    elif args.action == "quantize":
        # ONNX 모델 양자화
        optimizer = ONNXOptimizer(config)
        output_path = args.output_path or args.model_path.replace(
            ".onnx", "_quantized.onnx"
        )
        optimizer.quantize_model(args.model_path, output_path)

    elif args.action == "validate":
        # ONNX 모델 검증
        validator = ONNXValidator(config)
        model_info = validator.validate_model(args.model_path)
        if model_info:
            print("모델 정보:")
            for key, value in model_info.items():
                print(f"  {key}: {value}")

    elif args.action == "test":
        # ONNX 모델 추론 테스트
        validator = ONNXValidator(config)
        validator.test_inference(args.model_path, input_shape)


if __name__ == "__main__":
    main()
