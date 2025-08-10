#!/usr/bin/env python3
"""
PyTorch 모델을 ONNX로 변환하는 스크립트

Attention MIL 모델을 ONNX 형식으로 변환하여 프로덕션 배포에 최적화합니다.
"""

import argparse
import os
import sys
import torch
import onnx
import onnxruntime
import numpy as np
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.attention_mil import AttentionMILModel
from src.models.feature_extractor import FeatureExtractor
from src.utils.config import load_config


def create_dummy_input(batch_size: int = 1, num_tiles: int = 50, channels: int = 3, height: int = 256, width: int = 256):
    """
    ONNX 변환을 위한 더미 입력 생성
    
    Args:
        batch_size (int): 배치 크기
        num_tiles (int): 타일(프레임) 수
        channels (int): 채널 수
        height (int): 이미지 높이
        width (int): 이미지 너비
        
    Returns:
        torch.Tensor: 더미 입력 텐서
    """
    return torch.randn(batch_size, num_tiles, channels, height, width)


def convert_to_onnx(model: torch.nn.Module, 
                   dummy_input: torch.Tensor, 
                   output_path: str,
                   input_names: list = None,
                   output_names: list = None,
                   dynamic_axes: dict = None,
                   opset_version: int = 11):
    """
    PyTorch 모델을 ONNX로 변환
    
    Args:
        model (torch.nn.Module): 변환할 PyTorch 모델
        dummy_input (torch.Tensor): 더미 입력 텐서
        output_path (str): 출력 ONNX 파일 경로
        input_names (list): 입력 텐서 이름 리스트
        output_names (list): 출력 텐서 이름 리스트
        dynamic_axes (dict): 동적 축 설정
        opset_version (int): ONNX opset 버전
    """
    print(f"🔄 PyTorch 모델을 ONNX로 변환 중...")
    print(f"   - 입력 텐서 크기: {dummy_input.shape}")
    print(f"   - 출력 경로: {output_path}")
    print(f"   - ONNX opset 버전: {opset_version}")
    
    # 기본 입력/출력 이름 설정
    if input_names is None:
        input_names = ['input']
    if output_names is None:
        output_names = ['output']
    
    # 기본 동적 축 설정
    if dynamic_axes is None:
        dynamic_axes = {
            'input': {0: 'batch_size', 1: 'num_tiles'},
            'output': {0: 'batch_size'}
        }
    
    # 모델을 평가 모드로 설정
    model.eval()
    
    try:
        # ONNX 변환
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            verbose=False
        )
        
        print(f"✅ ONNX 변환 성공!")
        print(f"   - 파일 크기: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
        
    except Exception as e:
        print(f"❌ ONNX 변환 실패: {e}")
        raise


def validate_onnx_model(onnx_path: str, dummy_input: torch.Tensor):
    """
    변환된 ONNX 모델 검증
    
    Args:
        onnx_path (str): ONNX 모델 파일 경로
        dummy_input (torch.Tensor): 검증용 더미 입력
    """
    print(f"🔍 ONNX 모델 검증 중...")
    
    try:
        # ONNX 모델 로드 및 검증
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print(f"✅ ONNX 모델 형식 검증 성공")
        
        # ONNX Runtime으로 추론 테스트
        ort_session = onnxruntime.InferenceSession(onnx_path)
        
        # 입력 데이터 준비
        input_data = dummy_input.numpy()
        input_name = ort_session.get_inputs()[0].name
        
        # 추론 실행
        output = ort_session.run(None, {input_name: input_data})
        
        print(f"✅ ONNX Runtime 추론 테스트 성공")
        print(f"   - 출력 형태: {output[0].shape}")
        print(f"   - 출력 데이터 타입: {output[0].dtype}")
        
    except Exception as e:
        print(f"❌ ONNX 모델 검증 실패: {e}")
        raise


def load_pytorch_model(config_path: str, model_path: str):
    """
    PyTorch 모델 로드
    
    Args:
        config_path (str): 설정 파일 경로
        model_path (str): 모델 가중치 파일 경로
        
    Returns:
        torch.nn.Module: 로드된 모델
    """
    print(f"📥 PyTorch 모델 로드 중...")
    print(f"   - 설정 파일: {config_path}")
    print(f"   - 모델 가중치: {model_path}")
    
    # 설정 로드
    config = load_config(config_path)
    
    # 특징 추출기 생성
    feature_extractor = FeatureExtractor(
        model_name=config.get('feature_extractor', {}).get('model_name', 'efficientnet_b2'),
        pretrained=config.get('feature_extractor', {}).get('pretrained', False)
    )
    
    # Attention MIL 모델 생성
    model = AttentionMILModel(
        num_classes=config.get('model', {}).get('num_classes', 3),
        feature_extractor=feature_extractor,
        attention_hidden_dim=config.get('model', {}).get('attention_hidden_dim', 128),
        dropout_rate=config.get('model', {}).get('dropout_rate', 0.2)
    )
    
    # 모델 가중치 로드
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"✅ 모델 가중치 로드 성공")
    else:
        print(f"⚠️ 모델 가중치 파일이 없습니다. 초기화된 모델을 사용합니다.")
    
    return model


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='PyTorch 모델을 ONNX로 변환')
    parser.add_argument('--config', type=str, default='configs/model_configs/attention_mil.yaml',
                       help='모델 설정 파일 경로')
    parser.add_argument('--model_path', type=str, required=True,
                       help='PyTorch 모델 가중치 파일 경로 (.pth)')
    parser.add_argument('--output_path', type=str, default='models/attention_mil.onnx',
                       help='출력 ONNX 파일 경로')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='배치 크기')
    parser.add_argument('--num_tiles', type=int, default=50,
                       help='타일(프레임) 수')
    parser.add_argument('--height', type=int, default=256,
                       help='이미지 높이')
    parser.add_argument('--width', type=int, default=256,
                       help='이미지 너비')
    parser.add_argument('--opset_version', type=int, default=11,
                       help='ONNX opset 버전')
    parser.add_argument('--validate', action='store_true',
                       help='변환 후 ONNX 모델 검증 수행')
    
    args = parser.parse_args()
    
    print(f"🚀 PyTorch → ONNX 변환 시작")
    print(f"   - 설정 파일: {args.config}")
    print(f"   - 모델 경로: {args.model_path}")
    print(f"   - 출력 경로: {args.output_path}")
    
    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    try:
        # PyTorch 모델 로드
        model = load_pytorch_model(args.config, args.model_path)
        
        # 더미 입력 생성
        dummy_input = create_dummy_input(
            batch_size=args.batch_size,
            num_tiles=args.num_tiles,
            height=args.height,
            width=args.width
        )
        
        # ONNX 변환
        convert_to_onnx(
            model=model,
            dummy_input=dummy_input,
            output_path=args.output_path,
            opset_version=args.opset_version
        )
        
        # 검증 수행 (선택사항)
        if args.validate:
            validate_onnx_model(args.output_path, dummy_input)
        
        print(f"🎉 ONNX 변환 완료!")
        print(f"   - 출력 파일: {args.output_path}")
        
    except Exception as e:
        print(f"❌ 변환 실패: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 