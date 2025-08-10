#!/usr/bin/env python3
"""
ONNX 모델 최적화 스크립트

변환된 ONNX 모델을 최적화하여 추론 성능을 향상시킵니다.
"""

import argparse
import os
import sys
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def load_onnx_model(onnx_path: str):
    """
    ONNX 모델 로드
    
    Args:
        onnx_path (str): ONNX 모델 파일 경로
        
    Returns:
        onnx.ModelProto: 로드된 ONNX 모델
    """
    print(f"📥 ONNX 모델 로드 중...")
    print(f"   - 파일 경로: {onnx_path}")
    
    try:
        model = onnx.load(onnx_path)
        print(f"✅ ONNX 모델 로드 성공")
        print(f"   - 모델 버전: {model.ir_version}")
        print(f"   - Opset 버전: {model.opset_import[0].version}")
        print(f"   - 프로듀서: {model.producer_name}")
        print(f"   - 입력 수: {len(model.graph.input)}")
        print(f"   - 출력 수: {len(model.graph.output)}")
        print(f"   - 노드 수: {len(model.graph.node)}")
        
        return model
        
    except Exception as e:
        print(f"❌ ONNX 모델 로드 실패: {e}")
        raise


def analyze_onnx_model(model: onnx.ModelProto):
    """
    ONNX 모델 분석
    
    Args:
        model (onnx.ModelProto): 분석할 ONNX 모델
    """
    print(f"🔍 ONNX 모델 분석 중...")
    
    # 입력 정보
    print(f"   📥 입력 정보:")
    for i, input_info in enumerate(model.graph.input):
        print(f"     - 입력 {i}: {input_info.name}")
        if input_info.type.tensor_type.shape.dim:
            shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic' for dim in input_info.type.tensor_type.shape.dim]
            print(f"       형태: {shape}")
    
    # 출력 정보
    print(f"   📤 출력 정보:")
    for i, output_info in enumerate(model.graph.output):
        print(f"     - 출력 {i}: {output_info.name}")
        if output_info.type.tensor_type.shape.dim:
            shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic' for dim in output_info.type.tensor_type.shape.dim]
            print(f"       형태: {shape}")
    
    # 연산자 통계
    op_types = {}
    for node in model.graph.node:
        op_type = node.op_type
        op_types[op_type] = op_types.get(op_type, 0) + 1
    
    print(f"   ⚙️ 연산자 통계:")
    for op_type, count in sorted(op_types.items()):
        print(f"     - {op_type}: {count}개")


def optimize_onnx_model(input_path: str, output_path: str, optimization_level: str = 'all'):
    """
    ONNX 모델 최적화
    
    Args:
        input_path (str): 입력 ONNX 모델 파일 경로
        output_path (str): 출력 최적화된 ONNX 모델 파일 경로
        optimization_level (str): 최적화 레벨 ('basic', 'extended', 'all')
    """
    print(f"🔄 ONNX 모델 최적화 중...")
    print(f"   - 입력 파일: {input_path}")
    print(f"   - 출력 파일: {output_path}")
    print(f"   - 최적화 레벨: {optimization_level}")
    
    try:
        # ONNX 모델 로드
        model = load_onnx_model(input_path)
        
        # 모델 분석
        analyze_onnx_model(model)
        
        # 최적화 수행
        if optimization_level == 'basic':
            # 기본 최적화: 그래프 최적화
            print(f"   🔧 기본 최적화 수행 중...")
            optimized_model = onnx.optimizer.optimize(model)
            
        elif optimization_level == 'extended':
            # 확장 최적화: 추가 그래프 최적화
            print(f"   🔧 확장 최적화 수행 중...")
            optimized_model = onnx.optimizer.optimize(model)
            # 추가 최적화 옵션들
            optimized_model = onnx.shape_inference.infer_shapes(optimized_model)
            
        else:  # 'all'
            # 전체 최적화: 모든 최적화 기법 적용
            print(f"   🔧 전체 최적화 수행 중...")
            optimized_model = onnx.optimizer.optimize(model)
            optimized_model = onnx.shape_inference.infer_shapes(optimized_model)
            
            # 추가 최적화: 불필요한 노드 제거
            optimized_model = onnx.optimizer.optimize(optimized_model, ['eliminate_identity'])
        
        # 최적화된 모델 저장
        onnx.save(optimized_model, output_path)
        
        # 최적화 결과 분석
        print(f"✅ 최적화 완료!")
        print(f"   - 원본 파일 크기: {os.path.getsize(input_path) / (1024*1024):.2f} MB")
        print(f"   - 최적화 파일 크기: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
        
        size_reduction = (1 - os.path.getsize(output_path) / os.path.getsize(input_path)) * 100
        print(f"   - 크기 감소: {size_reduction:.1f}%")
        
        # 최적화된 모델 분석
        print(f"🔍 최적화된 모델 분석:")
        analyze_onnx_model(optimized_model)
        
    except Exception as e:
        print(f"❌ ONNX 모델 최적화 실패: {e}")
        raise


def benchmark_onnx_model(onnx_path: str, num_runs: int = 100):
    """
    ONNX 모델 성능 벤치마크
    
    Args:
        onnx_path (str): ONNX 모델 파일 경로
        num_runs (int): 벤치마크 실행 횟수
    """
    print(f"⚡ ONNX 모델 성능 벤치마크 중...")
    print(f"   - 모델 파일: {onnx_path}")
    print(f"   - 실행 횟수: {num_runs}")
    
    try:
        # ONNX Runtime 세션 생성
        session = ort.InferenceSession(onnx_path)
        
        # 입력 정보 가져오기
        input_info = session.get_inputs()[0]
        input_name = input_info.name
        input_shape = input_info.shape
        
        print(f"   📥 입력 정보:")
        print(f"     - 이름: {input_name}")
        print(f"     - 형태: {input_shape}")
        
        # 더미 입력 데이터 생성
        if input_shape[0] == 0:  # 동적 배치 크기
            input_shape = (1,) + tuple(input_shape[1:])
        
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        
        # 워밍업
        print(f"   🔥 워밍업 중...")
        for _ in range(10):
            session.run(None, {input_name: dummy_input})
        
        # 벤치마크 실행
        print(f"   🏃 벤치마크 실행 중...")
        import time
        
        times = []
        for i in range(num_runs):
            start_time = time.time()
            session.run(None, {input_name: dummy_input})
            end_time = time.time()
            times.append(end_time - start_time)
            
            if (i + 1) % 20 == 0:
                print(f"     - 진행률: {i + 1}/{num_runs}")
        
        # 결과 분석
        times = np.array(times)
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        print(f"✅ 벤치마크 완료!")
        print(f"   📊 성능 결과:")
        print(f"     - 평균 추론 시간: {avg_time*1000:.2f} ms")
        print(f"     - 표준편차: {std_time*1000:.2f} ms")
        print(f"     - 최소 시간: {min_time*1000:.2f} ms")
        print(f"     - 최대 시간: {max_time*1000:.2f} ms")
        print(f"     - FPS: {1/avg_time:.1f}")
        
    except Exception as e:
        print(f"❌ 벤치마크 실패: {e}")
        raise


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='ONNX 모델 최적화')
    parser.add_argument('--input_path', type=str, required=True,
                       help='입력 ONNX 모델 파일 경로')
    parser.add_argument('--output_path', type=str, required=True,
                       help='출력 최적화된 ONNX 모델 파일 경로')
    parser.add_argument('--optimization_level', type=str, default='all',
                       choices=['basic', 'extended', 'all'],
                       help='최적화 레벨 (기본값: all)')
    parser.add_argument('--benchmark', action='store_true',
                       help='최적화 후 성능 벤치마크 수행')
    parser.add_argument('--num_runs', type=int, default=100,
                       help='벤치마크 실행 횟수 (기본값: 100)')
    
    args = parser.parse_args()
    
    print(f"🚀 ONNX 모델 최적화 시작")
    print(f"   - 입력 파일: {args.input_path}")
    print(f"   - 출력 파일: {args.output_path}")
    print(f"   - 최적화 레벨: {args.optimization_level}")
    
    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    try:
        # ONNX 모델 최적화
        optimize_onnx_model(
            input_path=args.input_path,
            output_path=args.output_path,
            optimization_level=args.optimization_level
        )
        
        # 성능 벤치마크 (선택사항)
        if args.benchmark:
            print(f"\n" + "="*50)
            print(f"📊 원본 모델 벤치마크")
            print(f"="*50)
            benchmark_onnx_model(args.input_path, args.num_runs)
            
            print(f"\n" + "="*50)
            print(f"📊 최적화된 모델 벤치마크")
            print(f"="*50)
            benchmark_onnx_model(args.output_path, args.num_runs)
        
        print(f"🎉 ONNX 모델 최적화 완료!")
        print(f"   - 최적화된 파일: {args.output_path}")
        
    except Exception as e:
        print(f"❌ 최적화 실패: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 