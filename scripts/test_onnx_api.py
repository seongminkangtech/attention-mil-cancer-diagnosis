#!/usr/bin/env python3
"""
ONNX 모델 API 테스트 스크립트

FastAPI 서버가 ONNX 모델을 올바르게 사용하는지 테스트합니다.
"""

import requests
import json
import time
import os
from pathlib import Path

# API 서버 설정
API_BASE_URL = "http://localhost:8000"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"
PREDICT_ENDPOINT = f"{API_BASE_URL}/predict"
MODEL_INFO_ENDPOINT = f"{API_BASE_URL}/predict/model-info"


def test_health_check():
    """헬스 체크 테스트"""
    print("🔍 헬스 체크 테스트...")
    
    try:
        response = requests.get(HEALTH_ENDPOINT)
        if response.status_code == 200:
            print("✅ 헬스 체크 성공")
            return True
        else:
            print(f"❌ 헬스 체크 실패: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 헬스 체크 오류: {e}")
        return False


def test_model_info():
    """모델 정보 조회 테스트"""
    print("\n🔍 모델 정보 조회 테스트...")
    
    try:
        response = requests.get(MODEL_INFO_ENDPOINT)
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                model_info = data.get('data', {})
                print("✅ 모델 정보 조회 성공")
                print(f"   - 모델 타입: {model_info.get('model_type', 'N/A')}")
                print(f"   - 클래스 수: {model_info.get('num_classes', 'N/A')}")
                print(f"   - 모델 로드 상태: {model_info.get('model_loaded', 'N/A')}")
                print(f"   - ONNX 제공자: {model_info.get('providers', 'N/A')}")
                return True
            else:
                print(f"❌ 모델 정보 조회 실패: {data.get('message', 'N/A')}")
                return False
        else:
            print(f"❌ 모델 정보 조회 HTTP 오류: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 모델 정보 조회 오류: {e}")
        return False


def test_prediction():
    """추론 테스트"""
    print("\n🔍 추론 테스트...")
    
    # 샘플 이미지 경로 (테스트용)
    sample_images = [
        "tests/fixtures/sample_images/sample1.jpg",
        "tests/fixtures/sample_images/sample2.jpg"
    ]
    
    # 실제 이미지가 있는지 확인
    available_images = []
    for img_path in sample_images:
        if os.path.exists(img_path):
            available_images.append(img_path)
    
    if not available_images:
        print("⚠️ 테스트용 이미지를 찾을 수 없습니다. 더미 데이터로 테스트합니다.")
        # 더미 이미지 경로 생성
        available_images = ["dummy_image1.jpg", "dummy_image2.jpg"]
    
    # 추론 요청 데이터
    request_data = {
        "image_paths": available_images,
        "confidence_threshold": 0.5,
        "return_attention": True,
        "batch_size": 1
    }
    
    try:
        print(f"   - 입력 이미지: {available_images}")
        
        response = requests.post(PREDICT_ENDPOINT, json=request_data)
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                prediction_data = data.get('data', {})
                print("✅ 추론 성공")
                print(f"   - 예측 결과: {prediction_data.get('prediction', 'N/A')}")
                print(f"   - 신뢰도: {prediction_data.get('confidence', 'N/A'):.4f}")
                print(f"   - 처리 시간: {prediction_data.get('processing_time', 'N/A'):.4f}초")
                print(f"   - 입력 이미지 수: {prediction_data.get('input_image_count', 'N/A')}")
                
                # 클래스별 확률 출력
                class_probs = prediction_data.get('class_probabilities', {})
                if class_probs:
                    print("   - 클래스별 확률:")
                    for class_name, prob in class_probs.items():
                        print(f"     * {class_name}: {prob:.4f}")
                
                return True
            else:
                print(f"❌ 추론 실패: {data.get('message', 'N/A')}")
                return False
        else:
            print(f"❌ 추론 HTTP 오류: {response.status_code}")
            print(f"   - 응답: {response.text}")
            return False
    except Exception as e:
        print(f"❌ 추론 오류: {e}")
        return False


def test_model_reload():
    """모델 재로드 테스트"""
    print("\n🔍 모델 재로드 테스트...")
    
    try:
        response = requests.post(f"{API_BASE_URL}/predict/reload-model")
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print("✅ 모델 재로드 성공")
                return True
            else:
                print(f"❌ 모델 재로드 실패: {data.get('message', 'N/A')}")
                return False
        else:
            print(f"❌ 모델 재로드 HTTP 오류: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 모델 재로드 오류: {e}")
        return False


def main():
    """메인 테스트 함수"""
    print("🚀 ONNX 모델 API 테스트 시작")
    print(f"   - API 서버: {API_BASE_URL}")
    print(f"   - 테스트 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 테스트 결과 저장
    test_results = []
    
    # 1. 헬스 체크
    test_results.append(("헬스 체크", test_health_check()))
    
    # 2. 모델 정보 조회
    test_results.append(("모델 정보 조회", test_model_info()))
    
    # 3. 추론 테스트
    test_results.append(("추론 테스트", test_prediction()))
    
    # 4. 모델 재로드
    test_results.append(("모델 재로드", test_model_reload()))
    
    # 5. 최종 결과 출력
    print("\n" + "="*50)
    print("📊 테스트 결과 요약")
    print("="*50)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 통과" if result else "❌ 실패"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n전체: {passed}/{total} 통과")
    
    if passed == total:
        print("🎉 모든 테스트가 통과했습니다!")
        return 0
    else:
        print("⚠️ 일부 테스트가 실패했습니다.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code) 